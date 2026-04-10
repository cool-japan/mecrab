//! WikidataProcessor, BuildConfig, BuildProgress, and BuildResult
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Contains:
//! - `BuildConfig`    – pipeline configuration
//! - `BuildProgress`  – progress callback payload
//! - `BuildResult`    – statistics returned from the build
//! - `WikidataProcessor` – orchestrates the full build pipeline

use crate::{BuildError, Result, create_spinner};
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use super::index::WikidataIndex;
use super::parser::{
    parse_wikidata_dump, parse_wikidata_dump_streaming, pos_to_allowed_entity_types,
};

// ─────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for dictionary building
#[derive(Debug, Clone)]
pub struct BuildConfig {
    /// Source dictionary CSV path
    pub source_csv: PathBuf,
    /// Wikidata JSON dump path (optional)
    pub wikidata_path: Option<PathBuf>,
    /// Wikipedia abstract dump path (optional)
    pub wikipedia_path: Option<PathBuf>,
    /// Output directory
    pub output_dir: PathBuf,
    /// Maximum semantic candidates per word
    pub max_candidates: u8,
    /// Number of parallel workers (0 = auto)
    pub num_workers: usize,
    /// Verbose output
    pub verbose: bool,
    /// Wikidata entity type filter (Q-IDs to keep, empty = keep all)
    /// Example: ["Q5", "Q515"] keeps only persons and cities
    pub entity_type_filter: Vec<String>,
    /// Enable post-processing confidence calibration (ambiguity penalty)
    pub calibration_enabled: bool,

    /// Load existing index for incremental updates (optional).
    /// When set, the existing index is loaded before processing new data (delta mode).
    pub load_existing: Option<PathBuf>,

    /// Use memory-efficient bounded-channel streaming for very large dumps (> 10 GB).
    pub streaming: bool,

    /// DBpedia NTriples dump path (optional).
    ///
    /// When set, the DBpedia `.nt` (or `.nt.gz`) file is processed after the
    /// Wikidata phase and its surface→URI mappings are merged into the index.
    pub dbpedia_path: Option<PathBuf>,

    /// Enable online entity resolution via the Wikidata API.
    ///
    /// When `true`, after all local data sources have been processed, any
    /// proper-noun surface form that still has no semantic URI is looked up
    /// via the live Wikidata `wbsearchentities` API.
    pub online_resolution: bool,

    /// Custom ontology import paths (CSV, JSON, or RDF/OWL).
    ///
    /// These are processed in Phase 0 (before Wikidata/Wikipedia), giving
    /// user-provided URIs the highest priority during max-confidence merge.
    /// Supports `.csv`, `.json`, `.rdf`, `.owl`, and `.xml` extensions.
    pub ontology_paths: Vec<PathBuf>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            source_csv: PathBuf::new(),
            wikidata_path: None,
            wikipedia_path: None,
            output_dir: PathBuf::from("./output"),
            max_candidates: 5,
            num_workers: 0,
            verbose: false,
            entity_type_filter: Vec::new(),
            calibration_enabled: true,
            load_existing: None,
            streaming: false,
            dbpedia_path: None,
            online_resolution: false,
            ontology_paths: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Progress / Result types
// ─────────────────────────────────────────────────────────────

/// Build progress callback
#[derive(Debug, Clone)]
pub struct BuildProgress {
    /// Current phase
    pub phase: String,
    /// Items processed
    pub processed: u64,
    /// Total items (0 if unknown)
    pub total: u64,
}

/// Build result with statistics
#[derive(Debug, Clone)]
pub struct BuildResult {
    /// Number of entries processed
    pub entries_processed: u64,
    /// Number of entries with semantic URIs
    pub entries_with_semantics: u64,
    /// Total semantic candidates added
    pub total_candidates: u64,
    /// Output files created
    pub output_files: Vec<PathBuf>,
}

// ─────────────────────────────────────────────────────────────
// WikidataProcessor
// ─────────────────────────────────────────────────────────────

/// Wikidata processor for building semantic dictionaries
pub struct WikidataProcessor {
    config: BuildConfig,
    index: WikidataIndex,
}

impl WikidataProcessor {
    /// Create a new processor with the given configuration
    pub fn new(config: BuildConfig) -> Result<Self> {
        if !config.source_csv.exists() {
            return Err(BuildError::InvalidInput(format!(
                "Source CSV not found: {:?}",
                config.source_csv
            )));
        }

        if config.wikidata_path.is_none() && config.wikipedia_path.is_none() {
            return Err(BuildError::InvalidInput(
                "At least one of wikidata_path or wikipedia_path must be specified".to_string(),
            ));
        }

        Ok(Self {
            config,
            index: WikidataIndex::new(),
        })
    }

    /// Run the build pipeline
    pub async fn run(mut self) -> Result<BuildResult> {
        let mut result = BuildResult {
            entries_processed: 0,
            entries_with_semantics: 0,
            total_candidates: 0,
            output_files: Vec::new(),
        };

        // Configure rayon thread pool if requested
        if self.config.num_workers > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.config.num_workers)
                .build_global()
                .ok(); // ignore error if already initialised
        }

        // Load existing index if provided (incremental update / delta mode)
        if let Some(ref existing_path) = self.config.load_existing.clone() {
            if existing_path.exists() {
                match WikidataIndex::load_from_file(existing_path) {
                    Ok(loaded) => {
                        self.index = loaded;
                    }
                    Err(e) => {
                        // Non-fatal: log and continue with empty index
                        eprintln!("Warning: could not load existing index: {}", e);
                    }
                }
            }
        }

        // Phase 0: Load custom ontology data (highest priority, processed before all other sources)
        for ontology_path in self.config.ontology_paths.clone() {
            match crate::ontology::import_ontology(&ontology_path, &mut self.index, 0.8) {
                Ok(stats) => {
                    if self.config.verbose {
                        eprintln!(
                            "Ontology {}: {} surfaces added, {} aliases added",
                            ontology_path.display(),
                            stats.surfaces_added,
                            stats.aliases_added,
                        );
                    }
                }
                Err(e) => eprintln!(
                    "Warning: ontology import failed for {}: {}",
                    ontology_path.display(),
                    e
                ),
            }
        }

        // Phase 1: Build index from Wikidata
        if let Some(wikidata_path) = self.config.wikidata_path.clone() {
            if self.config.streaming {
                parse_wikidata_dump_streaming(
                    &wikidata_path,
                    &self.config.entity_type_filter,
                    self.config.calibration_enabled,
                    self.config.verbose,
                    100_000,
                    &mut self.index,
                )?;
            } else {
                parse_wikidata_dump(
                    &wikidata_path,
                    &self.config.entity_type_filter,
                    self.config.calibration_enabled,
                    &mut self.index,
                )?;
            }
        }

        // Phase 1a: Build index from Wikipedia abstracts (if provided)
        if let Some(wikipedia_path) = self.config.wikipedia_path.clone() {
            let wp = crate::wikipedia::WikipediaProcessor::new();
            match wp.process_dump(&wikipedia_path, &mut self.index, self.config.verbose) {
                Ok(stats) => {
                    if self.config.verbose {
                        eprintln!(
                            "Wikipedia: {} articles, {} surfaces added",
                            stats.articles_processed, stats.surfaces_added
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Wikipedia processing failed: {}", e);
                    // Non-fatal: continue without Wikipedia data
                }
            }
        }

        // Phase 1b: DBpedia NTriples data (if provided)
        if let Some(dbpedia_path) = self.config.dbpedia_path.clone() {
            let dp = crate::dbpedia::DBpediaProcessor::new();
            match dp.process_dump(&dbpedia_path, &mut self.index, self.config.verbose) {
                Ok(stats) => {
                    if self.config.verbose {
                        eprintln!(
                            "DBpedia: {} triples, {} labels, {} surfaces added",
                            stats.triples_processed, stats.labels_found, stats.surfaces_added
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Warning: DBpedia processing failed: {}", e);
                    // Non-fatal: continue without DBpedia data
                }
            }
        }

        // Phase 1c: Online entity resolution (Wikidata API fallback)
        if self.config.online_resolution {
            match crate::entity_resolver::EntityResolver::new() {
                Ok(mut resolver) => {
                    // Collect proper-noun surfaces that have no URI yet
                    let unresolved: Vec<String> = self
                        .index
                        .iter_entries()
                        .filter(|(_, uris)| uris.is_empty())
                        .map(|(s, _)| s.to_string())
                        .collect();

                    if !unresolved.is_empty() {
                        let surfaces: Vec<&str> = unresolved.iter().map(String::as_str).collect();
                        let added = resolver.resolve_missing(&surfaces, &mut self.index).await;
                        if self.config.verbose {
                            let (total_cached, non_empty) = resolver.cache_stats();
                            eprintln!(
                                "Online resolution: {} URIs added, {} cache entries ({} non-empty)",
                                added, total_cached, non_empty
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!(
                        "Warning: online entity resolver failed to initialise: {}",
                        e
                    );
                }
            }
        }

        // Phase 2: Merge with dictionary CSV
        let (processed, with_semantics, candidates) = self.merge_dictionary()?;
        result.entries_processed = processed;
        result.entries_with_semantics = with_semantics;
        result.total_candidates = candidates;

        result
            .output_files
            .push(self.config.output_dir.join("semantic.bin"));
        result
            .output_files
            .push(self.config.output_dir.join("extended.csv"));

        // Export index as CSV (for debugging/inspection)
        let csv_path = self.config.output_dir.join("semantic_index.csv");
        if let Ok(csv_file) = std::fs::File::create(&csv_path) {
            crate::csv_export::export_index_as_csv(&self.index, csv_file, 0.1).ok();
            result.output_files.push(csv_path);
        }

        Ok(result)
    }

    // ── Phase 2 ───────────────────────────────────────────────

    /// Merge index with dictionary CSV and output extended format
    fn merge_dictionary(&self) -> Result<(u64, u64, u64)> {
        let pb = create_spinner("Merging with dictionary...");

        let mut entries_processed = 0u64;
        let mut entries_with_semantics = 0u64;
        let mut total_candidates = 0u64;

        // Create output directory
        std::fs::create_dir_all(&self.config.output_dir)?;

        // Open source CSV
        let mut reader = csv::Reader::from_path(&self.config.source_csv)?;

        // Open output CSV
        let output_csv_path = self.config.output_dir.join("extended.csv");
        let mut writer = csv::Writer::from_path(&output_csv_path)?;

        // Build SemanticPool from index
        let pool_builder = self.index.to_semantic_pool();

        // Write semantic binary output
        let semantic_path = self.config.output_dir.join("semantic.bin");
        let mut semantic_file = File::create(&semantic_path)?;
        pool_builder.write_to(&mut semantic_file)?;

        // Write surface→URI mapping as JSON
        let mapping_path = self.config.output_dir.join("surface_map.json");
        let mut mapping_file = File::create(&mapping_path)?;
        // Serialise as { surface: [ [uri, conf], ... ] }
        let mapping: HashMap<&str, Vec<(String, f32)>> = self.index.iter_entries().collect();
        let mapping_json = serde_json::to_string(&mapping)?;
        std::io::Write::write_all(&mut mapping_file, mapping_json.as_bytes())?;

        // Process each row
        for row_result in reader.records() {
            let record = row_result?;
            entries_processed += 1;

            // Surface form is column 0
            let surface = record.get(0).unwrap_or("");

            // POS feature is column 4 (IPADIC layout)
            let pos_feature = record.get(4).unwrap_or("");
            let allowed_types = pos_to_allowed_entity_types(pos_feature);

            let mut semantic_ids: Vec<(String, f32)> = Vec::new();

            if let Some(uris) = self.index.lookup(surface) {
                let take_count = uris.len().min(self.config.max_candidates as usize);
                for (uri, confidence) in uris.into_iter().take(take_count) {
                    // POS-based entity type filter
                    // When `allowed_types` is Some([]) we skip all URIs.
                    // When it is None we allow all.
                    // When it is Some(ids) we would need entity-type data in
                    // the index to properly filter; for now we allow through
                    // (the filter was already applied during index building
                    // via `entity_type_filter` in BuildConfig).
                    match &allowed_types {
                        Some(ids) if ids.is_empty() => continue,
                        _ => {}
                    }
                    semantic_ids.push((uri, confidence));
                    total_candidates += 1;
                }
                if !semantic_ids.is_empty() {
                    entries_with_semantics += 1;
                }
            }

            // Write extended record
            let mut new_record: Vec<String> = record.iter().map(|s| s.to_string()).collect();

            if semantic_ids.is_empty() {
                new_record.push("0".to_string()); // count
                new_record.push(String::new()); // uri list (empty)
            } else {
                new_record.push(semantic_ids.len().to_string());
                new_record.push(
                    semantic_ids
                        .iter()
                        .map(|(uri, _)| uri.as_str())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }

            writer.write_record(&new_record)?;

            if entries_processed % 10_000 == 0 {
                pb.set_message(format!("Processed {} entries...", entries_processed));
            }
        }

        writer.flush()?;

        pb.finish_with_message(format!(
            "Done: {} entries, {} with semantics, {} candidates",
            entries_processed, entries_with_semantics, total_candidates
        ));

        Ok((entries_processed, entries_with_semantics, total_candidates))
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::parser::{
        Claim, ClaimValue, Claims, LabelValue, Mainsnak, SitelinkValue, WikidataEntry,
    };
    use super::*;
    use std::collections::HashMap;

    // ── Task 1: max-confidence dedup ─────────────────────────

    #[test]
    fn test_wikidata_index_basic() {
        let mut index = WikidataIndex::new();

        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("東京", "http://dbpedia.org/resource/Tokyo", 0.8);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);

        assert_eq!(index.len(), 2);
        assert_eq!(index.total_mappings(), 3);

        let tokyo = index.lookup("東京").unwrap();
        assert_eq!(tokyo.len(), 2);
        // Sorted by confidence descending
        assert!((tokyo[0].1 - 0.9).abs() < 1e-5);
        assert!((tokyo[1].1 - 0.8).abs() < 1e-5);

        let kyoto = index.lookup("京都").unwrap();
        assert_eq!(kyoto.len(), 1);
        assert!((kyoto[0].1 - 0.95).abs() < 1e-5);

        assert!(index.lookup("大阪").is_none());
    }

    #[test]
    fn test_wikidata_index_max_confidence_dedup() {
        let mut index = WikidataIndex::new();
        let uri = "http://www.wikidata.org/entity/Q1490";

        // Same URI added twice via label (0.9) and alias (0.7); must keep 0.9
        index.add("東京", uri, 0.9);
        index.add("東京", uri, 0.7);

        let results = index.lookup("東京").unwrap();
        assert_eq!(results.len(), 1, "Duplicate URI must be deduplicated");
        assert!(
            (results[0].1 - 0.9).abs() < 1e-5,
            "Must keep max confidence"
        );

        // A higher-confidence second add must win
        index.add("東京", uri, 0.95);
        let results = index.lookup("東京").unwrap();
        assert_eq!(results.len(), 1);
        assert!(
            (results[0].1 - 0.95).abs() < 1e-5,
            "Higher confidence must win"
        );
    }

    // ── Task 2: entity type filtering ───────────────────────

    #[test]
    fn test_entity_type_filter() {
        // Entry with P31 = Q5 (human)
        let entry = make_entry_with_types(vec!["Q5"]);
        assert!(entry.matches_type_filter(&[]));
        assert!(entry.matches_type_filter(&["Q5".to_string()]));
        assert!(!entry.matches_type_filter(&["Q515".to_string()])); // city

        // Entry with no claims
        let bare = WikidataEntry {
            id: "Q999".to_string(),
            labels: HashMap::new(),
            aliases: HashMap::new(),
            sitelinks: HashMap::new(),
            claims: Claims::default(),
        };
        assert!(bare.matches_type_filter(&[])); // empty filter = allow all
        assert!(!bare.matches_type_filter(&["Q5".to_string()]));
    }

    // ── Task 3: confidence calibration ──────────────────────

    #[test]
    fn test_calibrated_confidence_primary_boost() {
        let entry = WikidataEntry {
            id: "Q1490".to_string(),
            labels: {
                let mut m = HashMap::new();
                m.insert(
                    "ja".to_string(),
                    LabelValue {
                        language: "ja".to_string(),
                        value: "東京".to_string(),
                    },
                );
                m
            },
            aliases: HashMap::new(),
            sitelinks: HashMap::new(),
            claims: Claims::default(),
        };

        let primary = entry.calibrated_confidence("東京");
        let alias = entry.calibrated_confidence("東京都");
        assert!(
            primary > alias,
            "Primary label should have higher confidence than alias"
        );
    }

    #[test]
    fn test_recalibrate_ambiguity_penalty() {
        let mut index = WikidataIndex::new();
        // Two URIs for the same surface → ambiguity penalty 1/sqrt(2)
        index.add("東京", "http://www.wikidata.org/entity/Q1490", 1.0);
        index.add("東京", "http://dbpedia.org/resource/Tokyo", 1.0);

        index.recalibrate();

        let results = index.lookup("東京").unwrap();
        assert_eq!(results.len(), 2);
        let expected = (1.0_f32 / 2.0_f32.sqrt()).clamp(0.05, 1.0);
        for (_, conf) in &results {
            assert!(
                (conf - expected).abs() < 1e-5,
                "Expected {}, got {}",
                expected,
                conf
            );
        }

        // Single-URI surface should be unchanged
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.8);
        index.recalibrate();
        let kyoto = index.lookup("京都").unwrap();
        // After two recalibrate() calls with n=1: 0.8 * 1.0 = 0.8
        assert!((kyoto[0].1 - 0.8).abs() < 0.01);
    }

    // ── Original tests (must still pass) ─────────────────────

    #[test]
    fn test_to_semantic_pool() {
        let mut index = WikidataIndex::new();

        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("東京", "http://dbpedia.org/resource/Tokyo", 0.8);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);

        let pool_builder = index.to_semantic_pool();

        let mut buf = Vec::new();
        pool_builder.write_to(&mut buf).unwrap();

        assert!(buf.len() > 20);
        assert_eq!(&buf[0..4], b"MCSP");
    }

    #[test]
    fn test_semantic_pool_roundtrip() {
        use mecrab::semantic::pool::SemanticPool;

        let mut index = WikidataIndex::new();

        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);
        index.add("大阪", "http://dbpedia.org/resource/Osaka", 0.85);

        let pool_builder = index.to_semantic_pool();
        let mut buf = Vec::new();
        pool_builder.write_to(&mut buf).unwrap();

        let pool = SemanticPool::from_bytes(&buf).unwrap();

        assert_eq!(pool.len(), 3);
        assert!(!pool.is_empty());

        let uri1 = pool.get(1).unwrap();
        let uri2 = pool.get(2).unwrap();
        let uri3 = pool.get(3).unwrap();

        assert!(uri1.contains("wikidata.org") || uri1.contains("dbpedia.org"));
        assert!(uri2.contains("wikidata.org") || uri2.contains("dbpedia.org"));
        assert!(uri3.contains("wikidata.org") || uri3.contains("dbpedia.org"));

        assert!(pool.get_confidence(1).is_some());
        assert!(pool.get_confidence(2).is_some());
        assert!(pool.get_confidence(3).is_some());
    }

    #[test]
    fn test_wikidata_entry_japanese_surfaces() {
        let entry = WikidataEntry {
            id: "Q1490".to_string(),
            labels: {
                let mut map = HashMap::new();
                map.insert(
                    "ja".to_string(),
                    LabelValue {
                        language: "ja".to_string(),
                        value: "東京".to_string(),
                    },
                );
                map.insert(
                    "en".to_string(),
                    LabelValue {
                        language: "en".to_string(),
                        value: "Tokyo".to_string(),
                    },
                );
                map
            },
            aliases: {
                let mut map = HashMap::new();
                map.insert(
                    "ja".to_string(),
                    vec![
                        LabelValue {
                            language: "ja".to_string(),
                            value: "東京都".to_string(),
                        },
                        LabelValue {
                            language: "ja".to_string(),
                            value: "トウキョウ".to_string(),
                        },
                    ],
                );
                map
            },
            sitelinks: HashMap::new(),
            claims: Claims::default(),
        };

        let surfaces = entry.japanese_surfaces();
        assert_eq!(surfaces.len(), 3);
        assert!(surfaces.contains(&"東京".to_string()));
        assert!(surfaces.contains(&"東京都".to_string()));
        assert!(surfaces.contains(&"トウキョウ".to_string()));
    }

    #[test]
    fn test_wikidata_entry_popularity() {
        let mut entry = WikidataEntry {
            id: "Q1490".to_string(),
            labels: HashMap::new(),
            aliases: HashMap::new(),
            sitelinks: HashMap::new(),
            claims: Claims::default(),
        };

        assert!((entry.popularity() - 0.1).abs() < 0.001);

        for i in 0..50 {
            entry.sitelinks.insert(
                format!("site{}", i),
                SitelinkValue {
                    site: format!("site{}", i),
                    title: format!("title{}", i),
                },
            );
        }
        assert!((entry.popularity() - 0.6).abs() < 0.001);

        for i in 50..150 {
            entry.sitelinks.insert(
                format!("site{}", i),
                SitelinkValue {
                    site: format!("site{}", i),
                    title: format!("title{}", i),
                },
            );
        }
        assert!((entry.popularity() - 1.0).abs() < 0.001);
    }

    // ── Test helpers ─────────────────────────────────────────

    fn make_entry_with_types(type_qids: Vec<&str>) -> WikidataEntry {
        let instance_of = type_qids
            .into_iter()
            .map(|qid| Claim {
                mainsnak: Mainsnak {
                    datavalue: Some(ClaimValue {
                        value_type: Some("wikibase-entityid".to_string()),
                        value: Some(serde_json::json!({
                            "entity-type": "item",
                            "id": qid
                        })),
                    }),
                },
            })
            .collect();

        WikidataEntry {
            id: "Q42".to_string(),
            labels: HashMap::new(),
            aliases: HashMap::new(),
            sitelinks: HashMap::new(),
            claims: Claims { instance_of },
        }
    }
}
