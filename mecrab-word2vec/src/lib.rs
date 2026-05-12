//! mecrab-word2vec: Pure Rust Word2Vec implementation
//!
//! Fast, memory-efficient word2vec training optimized for Japanese morphological analysis.
//!
//! # Features
//!
//! - Skip-gram with negative sampling
//! - Multi-threaded training with Rayon
//! - Direct MCV1 format output
//! - Memory-efficient streaming
//! - FastText-style subword character n-gram embeddings
//!
//! # Example
//!
//! ```no_run
//! use mecrab_word2vec::Word2VecBuilder;
//!
//! let model = Word2VecBuilder::new()
//!     .vector_size(100)
//!     .window_size(5)
//!     .negative_samples(5)
//!     .min_count(10)
//!     .epochs(3)
//!     .threads(8)
//!     .build()?;
//!
//! model.train_from_file("corpus.txt")?;
//! model.save_text("vectors.txt")?;
//! # Ok::<(), anyhow::Error>(())
//! ```

mod io;
mod model;
pub mod subword;
mod skipgram;
mod trainer;
mod vocab;

pub use model::{SubwordConfig, Word2Vec, Word2VecBuilder};
pub use subword::CharNgramExtractor;
pub use vocab::Vocabulary;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Word2VecError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Vocabulary error: {0}")]
    Vocabulary(String),
}

pub type Result<T> = std::result::Result<T, Word2VecError>;

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Write;

    /// Write sentences to a temporary file and return its path.
    fn make_corpus_file(sentences: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("mecrab_w2v_test_{nanos}.txt"));
        let mut f = std::fs::File::create(&path).expect("create temp corpus file");
        for s in sentences {
            writeln!(f, "{s}").expect("write sentence");
        }
        path
    }

    /// Build a minimal corpus whose words (space-separated u32 IDs) all appear
    /// at least `min_count` times so vocabulary filtering keeps them.
    fn tiny_corpus() -> Vec<&'static str> {
        vec![
            "0 1 2 3 4",
            "1 2 3 4 5",
            "0 2 4 6 8",
            "1 3 5 7 9",
            "0 1 3 5 7",
            "2 4 6 8 0",
            "3 5 7 9 1",
            "0 3 6 9 2",
            "1 4 7 0 3",
            "2 5 8 1 4",
            "0 1 2 3 4",
            "1 2 3 4 5",
            "0 2 4 6 8",
            "1 3 5 7 9",
            "0 1 3 5 7",
        ]
    }

    // ── Happy-path: build vocab → create model → train → inspect ─────────────

    #[test]
    fn test_vocabulary_build_from_file() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        let mut vocab = Vocabulary::new(1, 0.0);
        vocab
            .build_from_file(&corpus_path)
            .expect("build_from_file should succeed");

        assert!(!vocab.is_empty(), "vocabulary must not be empty");
        assert!(
            vocab.len() >= 10,
            "tiny corpus has words 0-9, at least 10 entries expected"
        );
        assert!(vocab.total_words() > 0, "total word count must be positive");

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_word2vec_builder_and_train() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        // Build vocabulary and model via the builder
        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(3)
            .min_count(1)
            .sample(0.0) // no subsampling for tiny corpus
            .alpha(0.025)
            .min_alpha(0.0001)
            .epochs(2)
            .threads(1)
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus should succeed");

        // Check model dimensions match requested config
        let vector_size = model.config().vector_size;
        assert_eq!(vector_size, 8, "vector_size must match builder config");

        let vocab_size = model.vocab().len();
        assert!(
            vocab_size >= 10,
            "tiny corpus must produce at least 10 vocabulary entries"
        );

        // syn0 has shape [vocab_size, vector_size] stored flat
        assert_eq!(
            model.syn0.len(),
            vocab_size * vector_size,
            "syn0 length must equal vocab_size * vector_size"
        );

        // Verify at least one input vector is non-zero (initialised with random values)
        let any_non_zero = model.syn0.iter().any(|&v| v != 0.0);
        assert!(
            any_non_zero,
            "syn0 must contain non-zero values after initialisation"
        );

        // Train the model
        model
            .train_from_file(&corpus_path)
            .expect("train_from_file should succeed");

        // After training, syn0 should still have correct dimensions
        assert_eq!(model.syn0.len(), vocab_size * vector_size);

        let _ = std::fs::remove_file(&corpus_path);
    }

    // ── Save-and-reload round-trip via text format ────────────────────────────

    #[test]
    fn test_save_text_round_trip() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        let mut model = Word2VecBuilder::new()
            .vector_size(4)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus");

        model
            .train_from_file(&corpus_path)
            .expect("train_from_file");

        let out_path = std::env::temp_dir().join(format!(
            "mecrab_w2v_text_{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        model
            .save_text(&out_path)
            .expect("save_text should succeed");

        // Verify the file was created and has content
        let meta = std::fs::metadata(&out_path).expect("output file must exist after save_text");
        assert!(meta.len() > 0, "saved text file must be non-empty");

        // Read back and check header line: "<vocab_size> <vector_size>"
        let content = std::fs::read_to_string(&out_path).expect("read saved text file");
        let first_line = content
            .lines()
            .next()
            .expect("file must have at least one line");
        let parts: Vec<&str> = first_line.split_whitespace().collect();
        assert_eq!(parts.len(), 2, "header must have exactly two fields");

        let saved_vocab_size: usize = parts[0].parse().expect("first field must be vocab_size");
        let saved_vector_size: usize = parts[1].parse().expect("second field must be vector_size");

        assert_eq!(
            saved_vocab_size,
            model.vocab().len(),
            "saved vocab_size mismatch"
        );
        assert_eq!(
            saved_vector_size,
            model.config().vector_size,
            "saved vector_size mismatch"
        );

        let _ = std::fs::remove_file(&corpus_path);
        let _ = std::fs::remove_file(&out_path);
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_vocabulary_from_empty_corpus() {
        let corpus_path = make_corpus_file(&[]);

        let mut vocab = Vocabulary::new(1, 0.0);
        let result = vocab.build_from_file(&corpus_path);

        // Empty corpus should either succeed with empty vocab or return an error.
        // Either outcome is acceptable as long as it does not panic.
        if let Ok(()) = result {
            assert!(
                vocab.is_empty(),
                "empty corpus should produce empty vocabulary"
            );
        } // graceful error is also acceptable

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_builder_empty_corpus_returns_error() {
        let corpus_path = make_corpus_file(&[]);

        // build_from_corpus with empty corpus must fail gracefully (not panic)
        let result = Word2VecBuilder::new()
            .min_count(1)
            .build_from_corpus(&corpus_path);

        assert!(
            result.is_err(),
            "building a model from an empty corpus should return an error"
        );

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_training_config_defaults() {
        // Access default config via the builder (TrainingConfig is not re-exported)
        let dummy_sentences = tiny_corpus();
        let corpus_path = make_corpus_file(&dummy_sentences);

        let model = Word2VecBuilder::new()
            .min_count(1)
            .build_from_corpus(&corpus_path)
            .expect("build from tiny corpus");

        let cfg = model.config();
        assert!(cfg.vector_size > 0, "default vector_size must be positive");
        assert!(cfg.window_size > 0, "default window_size must be positive");
        assert!(cfg.epochs > 0, "default epochs must be positive");
        assert!(cfg.alpha > 0.0, "default alpha must be positive");
        assert!(cfg.min_alpha > 0.0, "default min_alpha must be positive");
        assert!(
            cfg.negative_samples > 0,
            "default negative_samples must be positive"
        );

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_vocabulary_contains_and_remapping() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        let mut vocab = Vocabulary::new(1, 0.0);
        vocab
            .build_from_file(&corpus_path)
            .expect("build_from_file");

        // All word IDs 0..=9 appear in the tiny corpus, so they must be in vocab.
        for word_id in 0u32..=9 {
            assert!(
                vocab.contains(word_id),
                "word_id {word_id} must be in vocabulary"
            );

            let info = vocab.get(word_id).expect("WordInfo must be present");
            assert_eq!(info.word_id, word_id, "word_id field must match key");
            assert!(info.count > 0, "count must be positive");

            // remapped_id must be a valid dense index
            let remapped = info.remapped_id as usize;
            assert!(remapped < vocab.len(), "remapped_id must be < vocab.len()");

            // round-trip: remapped_id → word_id
            let round_tripped = vocab
                .get_word_id(info.remapped_id)
                .expect("get_word_id must succeed for a valid remapped_id");
            assert_eq!(
                round_tripped, word_id,
                "remapped_id round-trip failed for word_id {word_id}"
            );
        }

        let _ = std::fs::remove_file(&corpus_path);
    }

    // ── Subword embedding tests ───────────────────────────────────────────────

    #[test]
    fn test_subword_config_defaults() {
        let cfg = SubwordConfig::default();
        assert_eq!(cfg.min_n, 3, "default min_n must be 3");
        assert_eq!(cfg.max_n, 6, "default max_n must be 6");
        assert_eq!(cfg.bucket_count, 2_000_000, "default bucket_count must be 2_000_000");
    }

    #[test]
    fn test_subword_extractor_basic() {
        let ext = CharNgramExtractor::new(2, 3, 100);
        let ids = ext.extract_bucket_ids("東京");
        // Should have multiple n-gram bucket IDs, all < 100
        assert!(!ids.is_empty(), "n-gram extraction must not be empty for non-trivial input");
        assert!(
            ids.iter().all(|&id| id < 100),
            "all bucket IDs must be in [0, bucket_count)"
        );
        // Should be sorted and deduplicated
        assert!(
            ids.windows(2).all(|w| w[0] <= w[1]),
            "bucket IDs must be sorted"
        );
        // All elements must be unique (deduplicated)
        let deduped: std::collections::HashSet<u32> = ids.iter().cloned().collect();
        assert_eq!(deduped.len(), ids.len(), "bucket IDs must be deduplicated");
    }

    #[test]
    fn test_subword_fasttext_oov_embedding() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        // Build a model with subword enabled using a small bucket_count for speed
        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            .with_subword(2, 3, 1000) // small bucket_count for test speed
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus with subword should succeed");

        // syn_ng must be allocated: bucket_count * vector_size
        assert_eq!(
            model.syn_ng.len(),
            1000 * 8,
            "syn_ng must have bucket_count * vector_size elements"
        );

        // Train the model
        model
            .train_from_file(&corpus_path)
            .expect("train_from_file with subword should succeed");

        // embed_surface on an OOV surface must return Some(vec) with at least one non-zero
        let oov_surface = "東京";
        let embedding = model.embed_surface(oov_surface);
        assert!(
            embedding.is_some(),
            "embed_surface must return Some for a non-trivial OOV surface"
        );
        let embedding = embedding.expect("already checked Some");
        assert_eq!(
            embedding.len(),
            8,
            "OOV embedding length must equal vector_size"
        );
        let any_non_zero = embedding.iter().any(|&v| v != 0.0);
        assert!(any_non_zero, "OOV embedding must have at least one non-zero value after training");

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_subword_model_without_subword_returns_none() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        let model = Word2VecBuilder::new()
            .vector_size(8)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            // No .with_subword() — subword disabled
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus");

        // embed_surface must return None when subword is not configured
        let result = model.embed_surface("東京");
        assert!(
            result.is_none(),
            "embed_surface must return None when subword is not configured"
        );

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_embed_word_with_subword_in_vocab() {
        let corpus_path = make_corpus_file(&tiny_corpus());

        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            .with_subword(2, 3, 1000)
            .build_from_corpus(&corpus_path)
            .expect("build_from_corpus");

        model
            .train_from_file(&corpus_path)
            .expect("train_from_file");

        // Word ID 0 is in vocabulary; look up its remapped ID
        let vocab = model.vocab();
        let remapped = vocab.get(0).map(|info| info.remapped_id);

        let embedding = model.embed_word_with_subword(remapped, "word_0_surface");
        assert!(
            embedding.is_some(),
            "embed_word_with_subword must return Some for an in-vocab word"
        );
        let embedding = embedding.expect("already checked Some");
        assert_eq!(embedding.len(), 8, "embedding length must equal vector_size");

        let _ = std::fs::remove_file(&corpus_path);
    }

    #[test]
    fn test_subword_save_load_roundtrip() {
        let corpus = make_corpus_file(&tiny_corpus());
        let surface_map: std::collections::HashMap<u32, String> = [
            (0u32, "東京".to_string()),
            (1u32, "は".to_string()),
            (2u32, "日本".to_string()),
            (3u32, "の".to_string()),
            (4u32, "首都".to_string()),
            (5u32, "です".to_string()),
            (6u32, "大阪".to_string()),
            (7u32, "京都".to_string()),
            (8u32, "神戸".to_string()),
            (9u32, "名古屋".to_string()),
        ]
        .into_iter()
        .collect();

        let mut model = Word2VecBuilder::new()
            .vector_size(8)
            .window_size(2)
            .negative_samples(2)
            .min_count(1)
            .sample(0.0)
            .epochs(1)
            .threads(1)
            .with_subword(2, 3, 1000) // small bucket count for test speed
            .build_from_corpus(&corpus)
            .expect("build_from_corpus");

        model.set_surface_map(surface_map);
        model.train_from_file(&corpus).expect("train_from_file");

        // Get embedding before save
        let before = model.embed_surface("テスト").expect("should embed");

        // Save subword table
        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let subword_path = dir.join(format!("mecrab_subword_test_{nanos}.subword"));
        model.save_subword_text(&subword_path).expect("save_subword_text");

        // Load into a fresh model (no subword config set — simulate inference deployment)
        let mut model2 = Word2VecBuilder::new()
            .vector_size(8)
            .min_count(1)
            .build_from_corpus(&corpus)
            .expect("build_from_corpus for model2");

        model2.load_subword_text(&subword_path).expect("load_subword_text");

        // Embedding after load should match
        let after = model2.embed_surface("テスト").expect("should embed after load");

        assert_eq!(before.len(), after.len(), "vector length must match");
        for (a, b) in before.iter().zip(after.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "values must be close after round-trip: {a} vs {b}"
            );
        }

        let _ = std::fs::remove_file(&corpus);
        let _ = std::fs::remove_file(&subword_path);
    }

    #[test]
    fn test_save_subword_no_op_when_disabled() {
        // Model without subword: save_subword_text should be a silent no-op
        let corpus = make_corpus_file(&tiny_corpus());
        let mut model = Word2VecBuilder::new()
            .vector_size(4)
            .min_count(1)
            .epochs(1)
            .threads(1)
            .build_from_corpus(&corpus)
            .expect("build");
        model.train_from_file(&corpus).expect("train");

        let dir = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("mecrab_subword_noop_{nanos}.subword"));
        // Should silently do nothing (no subword config set)
        model.save_subword_text(&path).expect("should succeed silently");
        // File should NOT be created for a non-subword model
        // (or it might be created empty — either behavior is acceptable)
        // Just verify no panic and returns Ok

        let _ = std::fs::remove_file(&corpus);
        let _ = std::fs::remove_file(&path); // might not exist — ignore error
    }
}
