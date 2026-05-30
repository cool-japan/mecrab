//! WikidataIndex: surface → URI mapping with confidence tracking
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Provides max-confidence deduplication, ambiguity-penalty recalibration,
//! merge, and save/load operations for the Wikidata surface→URI index.

use std::collections::HashMap;

use crate::BuildError;

// ─────────────────────────────────────────────────────────────
// WikidataIndex
// ─────────────────────────────────────────────────────────────

/// Index mapping surface forms to Wikidata URIs
///
/// Internally uses `HashMap<surface, HashMap<uri, max_confidence>>` to
/// guarantee O(1) deduplication with max-confidence semantics.
///
/// Entity-type data (P31 "instance of" Q-IDs per URI) is stored separately
/// so that [`WikidataProcessor`] can apply POS-based type filtering at
/// merge time even when the dump-level `entity_type_filter` was not set.
#[derive(Debug, Default)]
pub struct WikidataIndex {
    /// Surface form → (URI → max_confidence)
    pub(super) entries: HashMap<String, HashMap<String, f32>>,
    /// URI → P31 entity-type Q-IDs (e.g. ["Q5", "Q215380"]).
    ///
    /// This map is populated during dump parsing (Phase 1).  It is keyed by
    /// full URI (`http://www.wikidata.org/entity/Qxxx`) for consistency with
    /// the surface→URI entries above.  Entries may be absent for URIs sourced
    /// from Wikipedia/DBpedia or from indexes serialised before P31 storage
    /// was introduced.
    pub(super) entity_type_map: HashMap<String, Vec<String>>,
}

impl WikidataIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or update with max confidence) a surface → URI mapping
    pub fn add(&mut self, surface: &str, uri: &str, confidence: f32) {
        let uri_map = self.entries.entry(surface.to_string()).or_default();
        uri_map
            .entry(uri.to_string())
            .and_modify(|c| *c = c.max(confidence))
            .or_insert(confidence);
    }

    /// Store the P31 "instance of" entity-type Q-IDs for a URI.
    ///
    /// Called during dump parsing for every entry whose type list is non-empty.
    /// Existing types for the same URI are replaced (last write wins; in
    /// practice each URI appears only once in a dump).
    pub fn add_entity_types(&mut self, uri: &str, types: Vec<String>) {
        if !types.is_empty() {
            self.entity_type_map.insert(uri.to_string(), types);
        }
    }

    /// Return the P31 entity-type Q-IDs stored for `uri`, or an empty slice
    /// when no data is available.
    ///
    /// When the returned slice is empty the caller MUST treat the URI as
    /// passing any type filter (conservative / backward-compatible behaviour):
    ///
    /// ```text
    /// // entity_types empty (index predates P31 storage): allow through
    /// ```
    pub fn entity_types_for(&self, uri: &str) -> &[String] {
        self.entity_type_map
            .get(uri)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Look up URIs for a surface form, sorted by confidence descending
    pub fn lookup(&self, surface: &str) -> Option<Vec<(String, f32)>> {
        self.entries.get(surface).map(|uri_map| {
            let mut v: Vec<(String, f32)> = uri_map
                .iter()
                .map(|(uri, &conf)| (uri.clone(), conf))
                .collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        })
    }

    /// Get the number of unique surface forms
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get total number of URI mappings across all surfaces
    pub fn total_mappings(&self) -> usize {
        self.entries.values().map(|m| m.len()).sum()
    }

    /// Iterate over all entries as `(surface, sorted_uris)` pairs.
    /// Used for serialization (replaces the old `entries()` method).
    pub fn iter_entries(&self) -> impl Iterator<Item = (&str, Vec<(String, f32)>)> {
        self.entries.iter().map(|(surface, uri_map)| {
            let mut v: Vec<(String, f32)> = uri_map.iter().map(|(u, &c)| (u.clone(), c)).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            (surface.as_str(), v)
        })
    }

    /// Post-process confidences: apply 1/sqrt(n) ambiguity penalty where
    /// n = number of URI candidates for a given surface.
    ///
    /// Call after all entries have been added for accurate ambiguity counts.
    pub fn recalibrate(&mut self) {
        for uri_map in self.entries.values_mut() {
            let count = uri_map.len() as f32;
            // penalty = 1 / sqrt(n); for n=1 this is 1.0 (no change)
            let penalty = 1.0 / count.sqrt().max(1.0);
            for conf in uri_map.values_mut() {
                *conf = (*conf * penalty).clamp(0.05, 1.0);
            }
        }
    }

    /// Save index to a JSON file for incremental updates
    pub fn save_to_file(&self, path: &std::path::Path) -> std::result::Result<(), BuildError> {
        let mut map: std::collections::BTreeMap<&str, Vec<(&str, f32)>> =
            std::collections::BTreeMap::new();

        for (surface, _candidates) in self.iter_entries() {
            // Re-borrow from the inner HashMap directly to avoid the lifetime issue
            if let Some(uri_map) = self.entries.get(surface) {
                let mut v: Vec<(&str, f32)> =
                    uri_map.iter().map(|(u, &c)| (u.as_str(), c)).collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                map.insert(surface, v);
            }
        }

        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &map)?;
        Ok(())
    }

    /// Load index from a previously saved JSON file
    pub fn load_from_file(path: &std::path::Path) -> std::result::Result<Self, BuildError> {
        let file = std::fs::File::open(path)?;
        let map: std::collections::HashMap<String, Vec<(String, f32)>> =
            serde_json::from_reader(file)?;

        let mut index = WikidataIndex::default();
        for (surface, candidates) in map {
            for (uri, confidence) in candidates {
                index.add(&surface, &uri, confidence);
            }
        }
        Ok(index)
    }

    /// Remove all entries for a surface form.
    /// Returns `true` if the surface existed.
    pub fn remove_surface(&mut self, surface: &str) -> bool {
        self.entries.remove(surface).is_some()
    }

    /// Remove a specific URI for a surface form.
    /// Returns `true` if the URI existed.
    pub fn remove_uri(&mut self, surface: &str, uri: &str) -> bool {
        if let Some(uri_map) = self.entries.get_mut(surface) {
            let removed = uri_map.remove(uri).is_some();
            if uri_map.is_empty() {
                self.entries.remove(surface);
            }
            removed
        } else {
            false
        }
    }

    /// Merge another index into this one (keeping max confidence on conflicts)
    pub fn merge(&mut self, other: &WikidataIndex) {
        for (surface, uri_map) in &other.entries {
            let dest = self.entries.entry(surface.clone()).or_default();
            for (uri, &confidence) in uri_map {
                dest.entry(uri.clone())
                    .and_modify(|c| *c = c.max(confidence))
                    .or_insert(confidence);
            }
        }
        // Merge entity-type data (last-write-wins per URI is fine — each URI
        // appears in at most one Wikidata dump)
        for (uri, types) in &other.entity_type_map {
            self.entity_type_map
                .entry(uri.clone())
                .or_insert_with(|| types.clone());
        }
    }

    /// Count total unique surface forms (alias for `len`)
    pub fn surface_count(&self) -> usize {
        self.entries.len()
    }

    /// Convert index to SemanticPoolBuilder for binary serialization
    pub fn to_semantic_pool(&self) -> mecrab::semantic::pool::SemanticPoolBuilder {
        use mecrab::semantic::pool::{OntologySource, SemanticPoolBuilder};

        let mut builder = SemanticPoolBuilder::new();

        for uri_map in self.entries.values() {
            for (uri, &confidence) in uri_map {
                let source = if uri.contains("wikidata.org") {
                    OntologySource::Wikidata
                } else if uri.contains("dbpedia.org") {
                    OntologySource::DBpedia
                } else {
                    OntologySource::Custom
                };
                builder.add(uri, confidence, source);
            }
        }

        builder
    }
}
