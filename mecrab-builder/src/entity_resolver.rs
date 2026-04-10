//! Online entity resolution via the Wikidata REST API.
//!
//! Provides runtime fallback for surface forms not found in the
//! static semantic index, with in-memory caching to avoid redundant requests.
//!
//! # Rate Limiting
//!
//! The Wikidata API requests polite access. This implementation:
//! - Caches all results to minimise API calls
//! - Respects the User-Agent policy (sets a descriptive UA string)
//! - Does NOT implement hard rate limiting (callers should batch requests)

use crate::wikidata::WikidataIndex;
use crate::{BuildError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────

/// Maximum number of cached entries before the whole cache is evicted.
///
/// Using a simple evict-all strategy to avoid the overhead of a true LRU
/// while still bounding memory growth. For single-run pipeline usage this
/// limit is rarely reached.
const MAX_CACHE_ENTRIES: usize = 10_000;

// ─────────────────────────────────────────────────────────────
// API response types (private)
// ─────────────────────────────────────────────────────────────

/// Top-level response from `wbsearchentities`.
#[derive(Debug, Deserialize)]
struct WikidataSearchResponse {
    search: Vec<WikidataSearchResult>,
}

/// A single hit returned by the Wikidata search API.
#[derive(Debug, Deserialize)]
struct WikidataSearchResult {
    /// Wikidata item ID (e.g. `"Q1490"`).
    id: String,
    /// Display label in the requested language.
    #[serde(default)]
    label: String,
    /// Short description in the requested language.
    #[serde(default)]
    description: String,
    /// Match score returned by the API (higher = better; may be absent).
    #[serde(default)]
    score: f64,
}

// ─────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────

/// A resolved entity: canonical URI, human-readable metadata, and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedEntity {
    /// Wikidata entity URI, e.g. `"http://www.wikidata.org/entity/Q1490"`.
    pub uri: String,
    /// Label in the query language.
    pub label: String,
    /// Short description in the query language.
    pub description: String,
    /// Confidence score in `[0.0, 1.0]` (higher = more likely correct match).
    pub confidence: f32,
}

// ─────────────────────────────────────────────────────────────
// EntityResolver
// ─────────────────────────────────────────────────────────────

/// Online entity resolver with in-memory result cache.
///
/// Makes HTTP requests to the Wikidata `wbsearchentities` API to resolve
/// surface forms that are not found in the static semantic index.
///
/// # Caching
///
/// Uses a `HashMap`-based cache (not strict LRU). When the cache exceeds
/// [`MAX_CACHE_ENTRIES`] the entire cache is cleared to avoid unbounded memory
/// growth.  For typical pipeline usage (processing a single dictionary merge)
/// this eviction is never triggered.
pub struct EntityResolver {
    client: reqwest::Client,
    /// BCP-47 language code used for API queries (default: `"ja"`).
    pub language: String,
    /// Maximum number of API results per query (capped at 50 by Wikidata).
    pub limit: u8,
    /// Minimum confidence threshold — results below this are dropped.
    pub min_confidence: f32,
    /// In-memory cache: surface form → resolved entities.
    cache: HashMap<String, Vec<ResolvedEntity>>,
}

impl EntityResolver {
    /// Create a new entity resolver with default settings.
    ///
    /// Builds an [`reqwest::Client`] with:
    /// - A descriptive `User-Agent` header (required by Wikidata policy)
    /// - A 10-second request timeout
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Http`] if the HTTP client cannot be constructed.
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent(concat!(
                "MeCrab/",
                env!("CARGO_PKG_VERSION"),
                " (https://github.com/cool-japan/mecrab; semantic-enrichment-bot)",
            ))
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(BuildError::Http)?;

        Ok(Self {
            client,
            language: "ja".to_string(),
            limit: 5,
            min_confidence: 0.3,
            cache: HashMap::new(),
        })
    }

    /// Resolve a surface form to a list of [`ResolvedEntity`] values.
    ///
    /// Returns cached results when available; otherwise queries the Wikidata
    /// API.  Results are sorted by confidence descending.  On network error
    /// the empty list is cached and returned (avoids repeated failing calls).
    pub async fn resolve(&mut self, surface: &str) -> Vec<ResolvedEntity> {
        if let Some(cached) = self.cache.get(surface) {
            return cached.clone();
        }

        // Evict whole cache if it has grown too large
        if self.cache.len() >= MAX_CACHE_ENTRIES {
            self.cache.clear();
        }

        let results = match self.fetch_entities(surface).await {
            Ok(r) => r,
            Err(_) => {
                // Cache empty result to avoid retrying a failing surface
                self.cache.insert(surface.to_string(), Vec::new());
                return Vec::new();
            }
        };

        self.cache.insert(surface.to_string(), results.clone());
        results
    }

    /// Resolve a batch of surface forms, inserting new results into `index`.
    ///
    /// Only surfaces not already present in `index` are queried, minimising
    /// unnecessary API calls.
    ///
    /// Returns the number of (surface, URI) pairs inserted.
    pub async fn resolve_missing(&mut self, surfaces: &[&str], index: &mut WikidataIndex) -> u64 {
        let mut resolved_count = 0u64;

        for &surface in surfaces {
            if index.lookup(surface).is_some() {
                continue; // Already in index — skip
            }

            let entities = self.resolve(surface).await;
            for entity in &entities {
                if entity.confidence >= self.min_confidence {
                    index.add(surface, &entity.uri, entity.confidence);
                    resolved_count += 1;
                }
            }
        }

        resolved_count
    }

    /// Return cache statistics as `(total_entries, non_empty_entries)`.
    pub fn cache_stats(&self) -> (usize, usize) {
        let total = self.cache.len();
        let non_empty = self.cache.values().filter(|v| !v.is_empty()).count();
        (total, non_empty)
    }

    // ── Private ──────────────────────────────────────────────

    /// Issue a `wbsearchentities` request for `surface` and convert the
    /// response into a `Vec<ResolvedEntity>` sorted by confidence descending.
    async fn fetch_entities(&self, surface: &str) -> Result<Vec<ResolvedEntity>> {
        let url = format!(
            "https://www.wikidata.org/w/api.php\
             ?action=wbsearchentities\
             &search={}\
             &language={}\
             &format=json\
             &limit={}",
            urlencoding::encode(surface),
            self.language,
            self.limit,
        );

        let response: WikidataSearchResponse = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(BuildError::Http)?
            .json()
            .await
            .map_err(BuildError::Http)?;

        let mut resolved: Vec<ResolvedEntity> = response
            .search
            .into_iter()
            .enumerate()
            .filter_map(|(rank, result)| {
                // Rank-based score: 1st result → ~1.0, declining by 1/√(rank+1)
                let rank_score = 1.0_f64 / (rank as f64 + 1.0).sqrt();
                // API score normalised to [0, 1] (absent → neutral 0.5)
                let api_score_norm = if result.score > 0.0 {
                    (result.score / 100.0_f64).min(1.0)
                } else {
                    0.5
                };
                let confidence = ((rank_score + api_score_norm) / 2.0 * 0.8) as f32;

                if confidence < self.min_confidence {
                    return None;
                }

                Some(ResolvedEntity {
                    uri: format!("http://www.wikidata.org/entity/{}", result.id),
                    label: result.label,
                    description: result.description,
                    confidence,
                })
            })
            .collect();

        // Sort highest-confidence first
        resolved.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(resolved)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_encoding_japanese() {
        // Japanese characters must be percent-encoded in the URL
        let surface = "東京";
        let encoded = urlencoding::encode(surface).into_owned();
        // Each UTF-8 byte becomes %XX — the characters themselves should not
        // appear literally in the encoded form.
        assert!(!encoded.contains('東'), "Kanji should be percent-encoded");
        assert!(!encoded.contains('京'), "Kanji should be percent-encoded");
        assert!(
            encoded.starts_with('%'),
            "Should start with percent-encoded byte"
        );
    }

    #[test]
    fn test_resolved_entity_serde_roundtrip() {
        let entity = ResolvedEntity {
            uri: "http://www.wikidata.org/entity/Q1490".to_string(),
            label: "東京".to_string(),
            description: "日本の首都".to_string(),
            confidence: 0.8_f32,
        };
        let json = serde_json::to_string(&entity).expect("serialize");
        let back: ResolvedEntity = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.uri, entity.uri);
        assert_eq!(back.label, entity.label);
        assert_eq!(back.description, entity.description);
        assert!((back.confidence - entity.confidence).abs() < 1e-5);
    }

    #[test]
    fn test_confidence_below_threshold_filtered() {
        // Ensure items below min_confidence are dropped
        // We can't test fetch_entities directly without a network, but we can
        // verify the filter logic via a hand-crafted entity list.
        let entities: Vec<ResolvedEntity> = vec![
            ResolvedEntity {
                uri: "http://www.wikidata.org/entity/Q1".to_string(),
                label: "A".to_string(),
                description: String::new(),
                confidence: 0.1, // below default 0.3
            },
            ResolvedEntity {
                uri: "http://www.wikidata.org/entity/Q2".to_string(),
                label: "B".to_string(),
                description: String::new(),
                confidence: 0.5, // above threshold
            },
        ];

        let min = 0.3_f32;
        let filtered: Vec<&ResolvedEntity> =
            entities.iter().filter(|e| e.confidence >= min).collect();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].uri, "http://www.wikidata.org/entity/Q2");
    }

    #[test]
    fn test_cache_stats_empty() {
        let resolver = EntityResolver::new().expect("build resolver");
        let (total, non_empty) = resolver.cache_stats();
        assert_eq!(total, 0);
        assert_eq!(non_empty, 0);
    }

    #[tokio::test]
    #[ignore] // Requires live network access; run with --include-ignored
    async fn test_resolve_tokyo_live() {
        let mut resolver = EntityResolver::new().expect("build resolver");
        let results = resolver.resolve("東京").await;
        assert!(
            !results.is_empty(),
            "Should get at least one result for '東京'"
        );
        assert!(
            results[0].uri.contains("wikidata.org"),
            "URI should be a wikidata.org URL"
        );
        // First result confidence should be above threshold
        assert!(results[0].confidence >= resolver.min_confidence);
    }
}
