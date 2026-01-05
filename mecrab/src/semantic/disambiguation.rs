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
            // Use context for disambiguation (simplified)
            if let Some(ctx) = context {
                let mut scored: Vec<(f32, SemanticEntry)> = candidates
                    .iter()
                    .map(|e| {
                        let mut score = e.confidence;

                        // Boost if entity was already resolved in document
                        if ctx.resolved_entities.contains(&e.uri) {
                            score *= 1.5;
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
}
