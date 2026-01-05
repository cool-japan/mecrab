//! JSON-LD export for semantic analysis results

use super::SemanticEntry;

/// Escape a string for JSON
pub fn escape_json(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }
    result
}

/// Format a token as JSON-LD
pub fn format_token_jsonld(
    surface: &str,
    pos: &str,
    reading: Option<&str>,
    entities: &[SemanticEntry],
    pretty: bool,
) -> String {
    let indent = if pretty { "  " } else { "" };
    let newline = if pretty { "\n" } else { "" };

    let mut json = String::new();
    json.push('{');
    json.push_str(newline);

    // Context
    json.push_str(indent);
    json.push_str("\"@context\": {");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"wd\": \"http://www.wikidata.org/entity/\",");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"dbr\": \"http://dbpedia.org/resource/\",");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"schema\": \"http://schema.org/\"");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str("},");
    json.push_str(newline);

    // Surface
    json.push_str(indent);
    json.push_str(&format!("\"surface\": \"{}\",", escape_json(surface)));
    json.push_str(newline);

    // POS
    json.push_str(indent);
    json.push_str(&format!("\"pos\": \"{}\",", escape_json(pos)));
    json.push_str(newline);

    // Reading
    if let Some(reading) = reading {
        json.push_str(indent);
        json.push_str(&format!("\"reading\": \"{}\",", escape_json(reading)));
        json.push_str(newline);
    }

    // Entities
    json.push_str(indent);
    json.push_str("\"entities\": [");
    json.push_str(newline);

    for (i, entity) in entities.iter().enumerate() {
        json.push_str(indent);
        json.push_str(indent);
        json.push('{');
        json.push_str(newline);

        json.push_str(indent);
        json.push_str(indent);
        json.push_str(indent);
        json.push_str(&format!("\"@id\": \"{}\",", entity.uri));
        json.push_str(newline);

        json.push_str(indent);
        json.push_str(indent);
        json.push_str(indent);
        json.push_str(&format!("\"confidence\": {:.2}", entity.confidence));
        json.push_str(newline);

        json.push_str(indent);
        json.push_str(indent);
        json.push('}');
        if i < entities.len() - 1 {
            json.push(',');
        }
        json.push_str(newline);
    }

    json.push_str(indent);
    json.push(']');
    json.push_str(newline);

    json.push('}');
    json
}

/// Format multiple tokens as JSON-LD document
pub fn format_document_jsonld(
    tokens: &[(String, String, Option<String>, Vec<SemanticEntry>)],
    pretty: bool,
) -> String {
    let indent = if pretty { "  " } else { "" };
    let newline = if pretty { "\n" } else { "" };

    let mut json = String::new();
    json.push('{');
    json.push_str(newline);

    // Context
    json.push_str(indent);
    json.push_str("\"@context\": {");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"wd\": \"http://www.wikidata.org/entity/\",");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"dbr\": \"http://dbpedia.org/resource/\",");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"schema\": \"http://schema.org/\",");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str(indent);
    json.push_str("\"tokens\": \"@graph\"");
    json.push_str(newline);
    json.push_str(indent);
    json.push_str("},");
    json.push_str(newline);

    // Tokens
    json.push_str(indent);
    json.push_str("\"tokens\": [");
    json.push_str(newline);

    for (i, (surface, pos, reading, entities)) in tokens.iter().enumerate() {
        let token_json = format_token_jsonld(surface, pos, reading.as_deref(), entities, pretty);
        for line in token_json.lines() {
            json.push_str(indent);
            json.push_str(indent);
            json.push_str(line);
            json.push_str(newline);
        }
        if i < tokens.len() - 1 {
            // Remove last newline and add comma
            if json.ends_with(newline) {
                json.truncate(json.len() - newline.len());
            }
            json.push(',');
            json.push_str(newline);
        }
    }

    json.push_str(indent);
    json.push(']');
    json.push_str(newline);

    json.push('}');
    json
}

/// Check for entity deduplication across tokens
pub fn deduplicate_entities(entities: &mut Vec<SemanticEntry>) {
    let mut seen = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.uri.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::OntologySource;

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("hello\"world"), "hello\\\"world");
        assert_eq!(escape_json("line\nbreak"), "line\\nbreak");
    }

    #[test]
    fn test_format_basic() {
        let entities = vec![SemanticEntry::new(
            "http://www.wikidata.org/entity/Q1490",
            0.95,
            OntologySource::Wikidata,
        )];

        let json = format_token_jsonld("東京", "名詞", Some("トウキョウ"), &entities, false);
        assert!(json.contains("\"surface\": \"東京\""));
        assert!(json.contains("\"@id\": \"http://www.wikidata.org/entity/Q1490\""));
    }

    #[test]
    fn test_format_pretty() {
        let entities = vec![];
        let json = format_token_jsonld("テスト", "名詞", None, &entities, true);
        assert!(json.contains('\n'));
        assert!(json.contains("  "));
    }

    #[test]
    fn test_entity_deduplication() {
        let mut entities = vec![
            SemanticEntry::new("http://example.org/1", 0.9, OntologySource::Custom),
            SemanticEntry::new("http://example.org/1", 0.8, OntologySource::Custom),
            SemanticEntry::new("http://example.org/2", 0.7, OntologySource::Custom),
        ];

        deduplicate_entities(&mut entities);
        assert_eq!(entities.len(), 2);
    }
}
