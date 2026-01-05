//! Wikidata processing for semantic dictionary building
//!
//! This module handles:
//! - Streaming Wikidata JSON dumps
//! - Building surface → URI index
//! - Merging with dictionary CSV
//! - Outputting extended dictionary format

use crate::{BuildError, Result, create_spinner};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

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
        }
    }
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelValue {
    pub language: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SitelinkValue {
    pub site: String,
    pub title: String,
}

impl WikidataEntry {
    /// Get all Japanese surface forms for this entity
    pub fn japanese_surfaces(&self) -> Vec<String> {
        let mut surfaces = Vec::new();

        // Add Japanese label
        if let Some(label) = self.labels.get("ja") {
            surfaces.push(label.value.clone());
        }

        // Add Japanese aliases
        if let Some(aliases) = self.aliases.get("ja") {
            for alias in aliases {
                if !surfaces.contains(&alias.value) {
                    surfaces.push(alias.value.clone());
                }
            }
        }

        surfaces
    }

    /// Get popularity score based on sitelinks
    pub fn popularity(&self) -> f32 {
        let count = self.sitelinks.len();
        // Normalize: 0 sitelinks = 0.1, 100+ sitelinks = 1.0
        (0.1 + (count as f32 / 100.0).min(0.9)).min(1.0)
    }
}

/// Index mapping surface forms to Wikidata URIs
#[derive(Debug, Default)]
pub struct WikidataIndex {
    /// Surface form → Vec<(URI, confidence)>
    entries: HashMap<String, Vec<(String, f32)>>,
}

impl WikidataIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entry to the index
    pub fn add(&mut self, surface: &str, uri: &str, confidence: f32) {
        let entries = self.entries.entry(surface.to_string()).or_default();

        // Check if URI already exists
        if !entries.iter().any(|(u, _)| u == uri) {
            entries.push((uri.to_string(), confidence));
            // Sort by confidence descending
            entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    /// Look up URIs for a surface form
    pub fn lookup(&self, surface: &str) -> Option<&Vec<(String, f32)>> {
        self.entries.get(surface)
    }

    /// Get the number of unique surface forms
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total number of URI mappings
    pub fn total_mappings(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Convert index to SemanticPoolBuilder for binary serialization
    pub fn to_semantic_pool(&self) -> mecrab::semantic::pool::SemanticPoolBuilder {
        use mecrab::semantic::pool::{OntologySource, SemanticPoolBuilder};

        let mut builder = SemanticPoolBuilder::new();

        for uris in self.entries.values() {
            for (uri, confidence) in uris {
                // Determine source from URI prefix
                let source = if uri.contains("wikidata.org") {
                    OntologySource::Wikidata
                } else if uri.contains("dbpedia.org") {
                    OntologySource::DBpedia
                } else {
                    OntologySource::Custom
                };

                builder.add(uri, *confidence, source);
            }
        }

        builder
    }

    /// Get reference to entries for serialization
    pub fn entries(&self) -> &HashMap<String, Vec<(String, f32)>> {
        &self.entries
    }
}

/// Wikidata processor for building semantic dictionaries
pub struct WikidataProcessor {
    config: BuildConfig,
    index: WikidataIndex,
}

impl WikidataProcessor {
    /// Create a new processor with the given configuration
    pub fn new(config: BuildConfig) -> Result<Self> {
        // Validate config
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

        // Phase 1: Build index from Wikidata
        if let Some(wikidata_path) = self.config.wikidata_path.clone() {
            self.build_wikidata_index(&wikidata_path)?;
        }

        // Phase 2: Merge with dictionary CSV
        let (processed, with_semantics, candidates) = self.merge_dictionary()?;
        result.entries_processed = processed;
        result.entries_with_semantics = with_semantics;
        result.total_candidates = candidates;

        // Add output files
        result
            .output_files
            .push(self.config.output_dir.join("semantic.bin"));
        result
            .output_files
            .push(self.config.output_dir.join("extended.csv"));

        Ok(result)
    }

    /// Build the surface → URI index from Wikidata dump
    fn build_wikidata_index(&mut self, path: &Path) -> Result<()> {
        let pb = create_spinner("Loading Wikidata dump...");

        let file = File::open(path)?;
        let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(BufReader::new(GzDecoder::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };

        let mut count = 0u64;
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            // Skip array brackets
            if line == "[" || line == "]" {
                continue;
            }

            // Remove trailing comma if present
            let json_str = line.trim_end_matches(',');
            if json_str.is_empty() {
                continue;
            }

            // Parse entity
            if let Ok(entry) = serde_json::from_str::<WikidataEntry>(json_str) {
                let uri = format!("http://www.wikidata.org/entity/{}", entry.id);
                let confidence = entry.popularity();

                for surface in entry.japanese_surfaces() {
                    self.index.add(&surface, &uri, confidence);
                }

                count += 1;
                if count % 100_000 == 0 {
                    pb.set_message(format!("Processed {} entities...", count));
                }
            }
        }

        pb.finish_with_message(format!(
            "Indexed {} entities, {} surface forms, {} mappings",
            count,
            self.index.len(),
            self.index.total_mappings()
        ));

        Ok(())
    }

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

        // Write surface→URI mapping as JSON for now (TODO: binary format)
        let mapping_path = self.config.output_dir.join("surface_map.json");
        let mut mapping_file = File::create(&mapping_path)?;
        let mapping_json = serde_json::to_string(self.index.entries())?;
        std::io::Write::write_all(&mut mapping_file, mapping_json.as_bytes())?;

        // Process each row
        for result in reader.records() {
            let record = result?;
            entries_processed += 1;

            // Get surface form (first column)
            let surface = record.get(0).unwrap_or("");

            // Look up semantic URIs
            let mut semantic_ids = Vec::new();
            if let Some(uris) = self.index.lookup(surface) {
                let take_count = uris.len().min(self.config.max_candidates as usize);
                for (uri, confidence) in uris.iter().take(take_count) {
                    semantic_ids.push((uri.clone(), *confidence));
                    total_candidates += 1;
                }
                if !semantic_ids.is_empty() {
                    entries_with_semantics += 1;
                }
            }

            // Write extended record
            let mut new_record: Vec<String> = record.iter().map(|s| s.to_string()).collect();

            // Add semantic columns
            if semantic_ids.is_empty() {
                new_record.push("0".to_string()); // start_idx
                new_record.push("0".to_string()); // count
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wikidata_index_basic() {
        let mut index = WikidataIndex::new();

        // Add some entries
        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("東京", "http://dbpedia.org/resource/Tokyo", 0.8);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);

        assert_eq!(index.len(), 2);
        assert_eq!(index.total_mappings(), 3);

        // Test lookup
        let tokyo = index.lookup("東京").unwrap();
        assert_eq!(tokyo.len(), 2);
        // Should be sorted by confidence descending
        assert_eq!(tokyo[0].1, 0.9);
        assert_eq!(tokyo[1].1, 0.8);

        let kyoto = index.lookup("京都").unwrap();
        assert_eq!(kyoto.len(), 1);
        assert_eq!(kyoto[0].1, 0.95);

        assert!(index.lookup("大阪").is_none());
    }

    #[test]
    fn test_to_semantic_pool() {
        let mut index = WikidataIndex::new();

        // Add test entries
        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("東京", "http://dbpedia.org/resource/Tokyo", 0.8);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);

        // Convert to SemanticPool
        let pool_builder = index.to_semantic_pool();

        // The pool should contain all 3 URIs
        // We can't directly inspect the builder, but we can write it and verify the output
        let mut buf = Vec::new();
        pool_builder.write_to(&mut buf).unwrap();

        // Check that we have valid output
        assert!(buf.len() > 20); // At least header size
        assert_eq!(&buf[0..4], b"MCSP"); // Magic bytes
    }

    #[test]
    fn test_semantic_pool_roundtrip() {
        use mecrab::semantic::pool::SemanticPool;

        let mut index = WikidataIndex::new();

        // Add test entries with known URIs
        index.add("東京", "http://www.wikidata.org/entity/Q1490", 0.9);
        index.add("京都", "http://www.wikidata.org/entity/Q34600", 0.95);
        index.add("大阪", "http://dbpedia.org/resource/Osaka", 0.85);

        // Convert to SemanticPool and serialize
        let pool_builder = index.to_semantic_pool();
        let mut buf = Vec::new();
        pool_builder.write_to(&mut buf).unwrap();

        // Deserialize and verify
        let pool = SemanticPool::from_bytes(&buf).unwrap();

        // Check entry count
        assert_eq!(pool.len(), 3);

        // Verify we can read entries (we don't have direct access to URIs,
        // but we can verify the pool is valid and non-empty)
        assert!(!pool.is_empty());

        // Test that we can read back the URIs
        let uri1 = pool.get(1).unwrap();
        let uri2 = pool.get(2).unwrap();
        let uri3 = pool.get(3).unwrap();

        // Verify URIs contain the expected parts
        assert!(uri1.contains("wikidata.org") || uri1.contains("dbpedia.org"));
        assert!(uri2.contains("wikidata.org") || uri2.contains("dbpedia.org"));
        assert!(uri3.contains("wikidata.org") || uri3.contains("dbpedia.org"));

        // Check confidence values are preserved
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
        };

        // No sitelinks = 0.1
        assert!((entry.popularity() - 0.1).abs() < 0.001);

        // Add 50 sitelinks
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

        // Add 100+ sitelinks (should cap at 1.0)
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
}
