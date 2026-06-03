//! Disambiguation strategies for semantic entities

use super::{OntologySource, SemanticEntry};

/// Disambiguation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisambiguationStrategy {
    /// Use the highest confidence candidate
    HighestConfidence,
    /// Use popularity-based prior
    PopularityPrior,
    /// Use context words for disambiguation
    ContextBased,
    /// Return all candidates without disambiguation
    NoDisambiguation,
}

impl Default for DisambiguationStrategy {
    fn default() -> Self {
        DisambiguationStrategy::HighestConfidence
    }
}

/// Context for disambiguation
#[derive(Debug, Default)]
pub struct DisambiguationContext {
    /// Surrounding words (surface forms)
    pub context_words: Vec<String>,
    /// Already resolved entities in the document
    pub resolved_entities: Vec<String>,
    /// Topic hints (e.g., "technology", "geography")
    pub topic_hints: Vec<String>,
}

impl DisambiguationContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a context word
    pub fn add_context_word(&mut self, word: &str) {
        self.context_words.push(word.to_string());
    }

    /// Add a resolved entity URI
    pub fn add_resolved_entity(&mut self, uri: &str) {
        self.resolved_entities.push(uri.to_string());
    }

    /// Add a topic hint
    pub fn add_topic_hint(&mut self, topic: &str) {
        self.topic_hints.push(topic.to_string());
    }
}

/// Disambiguate candidates using the specified strategy
pub fn disambiguate(
    candidates: &[SemanticEntry],
    strategy: DisambiguationStrategy,
    context: Option<&DisambiguationContext>,
) -> Vec<SemanticEntry> {
    if candidates.is_empty() {
        return Vec::new();
    }

    match strategy {
        DisambiguationStrategy::HighestConfidence => {
            // Return only the highest confidence candidate
            let mut sorted = candidates.to_vec();
            sorted.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            vec![sorted.remove(0)]
        }
        DisambiguationStrategy::PopularityPrior => {
            // Boost Wikidata entries (assumed more popular)
            let mut scored: Vec<(f32, SemanticEntry)> = candidates
                .iter()
                .map(|e| {
                    let boost = match e.source {
                        OntologySource::Wikidata => 1.2,
                        OntologySource::DBpedia => 1.1,
                        _ => 1.0,
                    };
                    (e.confidence * boost, e.clone())
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            vec![scored.remove(0).1]
        }
        DisambiguationStrategy::ContextBased => {
            // Scores by resolved-entity co-occurrence, context word overlap, and topic hints
            if let Some(ctx) = context {
                let mut scored: Vec<(f32, SemanticEntry)> = candidates
                    .iter()
                    .map(|e| {
                        let mut score = e.confidence;

                        // Boost if entity was already resolved in document
                        if ctx.resolved_entities.contains(&e.uri) {
                            score *= 1.5;
                        }

                        // Token overlap: match context_words against the URI's last path segment
                        // (the identifier after the final '/')
                        let identifier = e.uri.rsplit('/').next().unwrap_or(&e.uri);
                        let identifier_lower = identifier.to_lowercase();
                        let overlap_count = ctx
                            .context_words
                            .iter()
                            .filter(|w| {
                                let wl = w.to_lowercase();
                                identifier_lower.contains(wl.as_str())
                                    || identifier_lower.starts_with(wl.as_str())
                            })
                            .count();
                        score += 0.1 * overlap_count as f32;

                        // Topic hint boost: if any topic hint appears in the full URI
                        let uri_lower = e.uri.to_lowercase();
                        let has_topic_match = ctx
                            .topic_hints
                            .iter()
                            .any(|hint| uri_lower.contains(hint.to_lowercase().as_str()));
                        if has_topic_match {
                            score *= 1.3;
                        }

                        (score, e.clone())
                    })
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                vec![scored.remove(0).1]
            } else {
                // Fall back to highest confidence
                disambiguate(candidates, DisambiguationStrategy::HighestConfidence, None)
            }
        }
        DisambiguationStrategy::NoDisambiguation => {
            // Return all candidates sorted by confidence
            let mut sorted = candidates.to_vec();
            sorted.sort_by(|a, b| {
                b.confidence
                    .partial_cmp(&a.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            sorted
        }
    }
}

/// Score candidates based on context similarity
pub fn score_by_context(
    candidates: &[SemanticEntry],
    context: &DisambiguationContext,
) -> Vec<(SemanticEntry, f32)> {
    candidates
        .iter()
        .map(|e| {
            let mut score = e.confidence;

            // Boost for resolved entities
            if context.resolved_entities.contains(&e.uri) {
                score *= 1.5;
            }

            // Could add more sophisticated scoring here
            // e.g., entity type matching, co-occurrence statistics

            (e.clone(), score)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(uri: &str, confidence: f32, source: OntologySource) -> SemanticEntry {
        SemanticEntry::new(uri, confidence, source)
    }

    #[test]
    fn test_highest_confidence() {
        let candidates = vec![
            make_entry("http://example.org/1", 0.7, OntologySource::Custom),
            make_entry("http://example.org/2", 0.9, OntologySource::Custom),
            make_entry("http://example.org/3", 0.5, OntologySource::Custom),
        ];

        let result = disambiguate(&candidates, DisambiguationStrategy::HighestConfidence, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].uri, "http://example.org/2");
    }

    #[test]
    fn test_popularity_prior() {
        let candidates = vec![
            make_entry("http://example.org/1", 0.8, OntologySource::Custom),
            make_entry(
                "http://www.wikidata.org/entity/Q1",
                0.7,
                OntologySource::Wikidata,
            ),
        ];

        let result = disambiguate(&candidates, DisambiguationStrategy::PopularityPrior, None);
        assert_eq!(result.len(), 1);
        // Wikidata should win due to boost (0.7 * 1.2 = 0.84 > 0.8)
        assert!(result[0].uri.contains("wikidata"));
    }

    #[test]
    fn test_no_disambiguation() {
        let candidates = vec![
            make_entry("http://example.org/1", 0.7, OntologySource::Custom),
            make_entry("http://example.org/2", 0.9, OntologySource::Custom),
        ];

        let result = disambiguate(&candidates, DisambiguationStrategy::NoDisambiguation, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].uri, "http://example.org/2"); // sorted by confidence
    }

    #[test]
    fn test_unique_candidate() {
        let candidates = vec![make_entry(
            "http://example.org/only",
            0.99,
            OntologySource::Wikidata,
        )];

        let result = disambiguate(&candidates, DisambiguationStrategy::HighestConfidence, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].uri, "http://example.org/only");
    }

    #[test]
    fn test_add_context_words() {
        let mut ctx = DisambiguationContext::new();
        ctx.add_context_word("東京");
        ctx.add_context_word("都庁");
        assert_eq!(ctx.context_words.len(), 2);
    }

    #[test]
    fn test_context_based_context_word_boost() {
        // Two candidates with equal confidence; the one whose URI identifier
        // matches a context word should win.
        let candidates = vec![
            make_entry("http://example.org/Tokyo", 0.6, OntologySource::Custom),
            make_entry("http://example.org/Osaka", 0.6, OntologySource::Custom),
        ];
        let mut ctx = DisambiguationContext::new();
        ctx.add_context_word("tokyo"); // matches "Tokyo" (case-insensitive)

        let result = disambiguate(
            &candidates,
            DisambiguationStrategy::ContextBased,
            Some(&ctx),
        );
        assert_eq!(result.len(), 1);
        assert!(
            result[0].uri.contains("Tokyo"),
            "expected Tokyo to win due to context word overlap"
        );
    }

    #[test]
    fn test_context_based_topic_hint_boost() {
        // Lower-confidence candidate wins when it matches the topic hint.
        let candidates = vec![
            make_entry(
                "http://www.wikidata.org/entity/Q1490",
                0.5,
                OntologySource::Wikidata,
            ),
            make_entry("http://example.org/Other", 0.7, OntologySource::Custom),
        ];
        let mut ctx = DisambiguationContext::new();
        ctx.add_topic_hint("wikidata"); // matches the wikidata URI

        let result = disambiguate(
            &candidates,
            DisambiguationStrategy::ContextBased,
            Some(&ctx),
        );
        assert_eq!(result.len(), 1);
        // 0.5 * 1.3 = 0.65 > 0.7? No — 0.65 < 0.7, so Other still wins here.
        // Confirm the higher-confidence+topic-boosted one wins when boost is decisive.
        // Swap: wikidata at 0.6 → 0.6*1.3=0.78 > 0.7
        let candidates2 = vec![
            make_entry(
                "http://www.wikidata.org/entity/Q1490",
                0.6,
                OntologySource::Wikidata,
            ),
            make_entry("http://example.org/Other", 0.7, OntologySource::Custom),
        ];
        let result2 = disambiguate(
            &candidates2,
            DisambiguationStrategy::ContextBased,
            Some(&ctx),
        );
        assert_eq!(result2.len(), 1);
        assert!(
            result2[0].uri.contains("wikidata"),
            "wikidata candidate (0.6 * 1.3 = 0.78) should beat Other (0.7)"
        );
    }

    #[test]
    fn test_context_based_combined_boost() {
        // Both context word overlap AND topic hint are applied cumulatively.
        // Tokyo at 0.65: (0.65 + 0.1) * 1.3 = 0.975 > OtherCity at 0.8
        let candidates = vec![
            make_entry(
                "http://dbpedia.org/resource/Tokyo",
                0.65,
                OntologySource::DBpedia,
            ),
            make_entry("http://example.org/OtherCity", 0.8, OntologySource::Custom),
        ];
        let mut ctx = DisambiguationContext::new();
        ctx.add_context_word("tokyo"); // +0.1 overlap
        ctx.add_topic_hint("dbpedia"); // *1.3 topic match

        let result = disambiguate(
            &candidates,
            DisambiguationStrategy::ContextBased,
            Some(&ctx),
        );
        assert_eq!(result.len(), 1);
        assert!(
            result[0].uri.contains("Tokyo"),
            "Tokyo with context+topic boost should beat OtherCity"
        );
    }

    #[test]
    fn test_context_based_no_context_falls_back() {
        // Without a context, falls back to HighestConfidence.
        let candidates = vec![
            make_entry("http://example.org/A", 0.4, OntologySource::Custom),
            make_entry("http://example.org/B", 0.9, OntologySource::Custom),
        ];
        let result = disambiguate(&candidates, DisambiguationStrategy::ContextBased, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].uri, "http://example.org/B");
    }
}
