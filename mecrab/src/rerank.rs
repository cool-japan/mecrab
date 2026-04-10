//! Neural reranking for N-best path selection.
//!
//! # Overview
//! After the Viterbi algorithm produces N candidate segmentations, a reranker
//! can select the most linguistically accurate path. The default [`NullReranker`]
//! always selects index 0 (the Viterbi-optimal path). Enable `--features neural`
//! for the [`NeuralReranker`] based on a lightweight BERT model.
//!
//! # Example
//! ```no_run
//! use mecrab::rerank::{NullReranker, Reranker, RerankCandidate};
//!
//! let reranker = NullReranker;
//! let candidates = vec![
//!     RerankCandidate {
//!         surfaces: vec!["東京".to_owned(), "は".to_owned()],
//!         pos_tags: vec!["名詞".to_owned(), "助詞".to_owned()],
//!         cost: 100,
//!     },
//! ];
//! let best = reranker.rerank(&candidates);
//! assert_eq!(best, 0);
//! ```

/// A single candidate path from N-best search.
///
/// Wraps the morpheme sequence extracted from an [`crate::AnalysisResult`]
/// together with the total Viterbi lattice cost for that path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankCandidate {
    /// Surface forms of morphemes in this path.
    pub surfaces: Vec<String>,
    /// Part-of-speech tags (first CSV field of each morpheme's feature string).
    pub pos_tags: Vec<String>,
    /// Total Viterbi path cost (lower = better).
    pub cost: i64,
}

/// Trait for reranking N-best candidates.
///
/// Implementations receive a slice of candidates and return the index of the
/// preferred candidate.  Thread-safety (`Send + Sync`) is required for use
/// in concurrent batch processing pipelines.
///
/// # Contract
///
/// - If `candidates` is empty the implementation **must** return `0`.
/// - The returned index **must** be in `0..candidates.len()` when
///   `!candidates.is_empty()`.
pub trait Reranker: Send + Sync {
    /// Select the best candidate. Returns an index into `candidates`.
    ///
    /// Must return `0` when `candidates` is empty.
    fn rerank(&self, candidates: &[RerankCandidate]) -> usize;

    /// Human-readable name for logging / debugging.
    fn name(&self) -> &'static str;
}

// ─── NullReranker ───────────────────────────────────────────────────────────

/// Default no-op reranker — always picks the Viterbi-optimal path (index 0).
///
/// This is the production default and adds **zero** overhead to N-best search.
/// Because the Viterbi algorithm already returns paths sorted by cost, index 0
/// is always the globally optimal path.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullReranker;

impl Reranker for NullReranker {
    #[inline]
    fn rerank(&self, _candidates: &[RerankCandidate]) -> usize {
        0
    }

    fn name(&self) -> &'static str {
        "NullReranker"
    }
}

// ─── CostReranker ────────────────────────────────────────────────────────────

/// Simple statistical reranker based on total path cost.
///
/// Scans all candidates and selects the one with the minimum [`RerankCandidate::cost`].
/// This produces the same result as `NullReranker` when candidates are already
/// sorted by the Viterbi solver, but can differ when a custom candidate list is
/// provided out-of-order.
///
/// Useful for testing the reranking pipeline without a neural model.
#[derive(Debug, Default, Clone, Copy)]
pub struct CostReranker;

impl Reranker for CostReranker {
    fn rerank(&self, candidates: &[RerankCandidate]) -> usize {
        candidates
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.cost)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn name(&self) -> &'static str {
        "CostReranker"
    }
}

// ─── Neural reranker (feature-gated) ─────────────────────────────────────────

#[cfg(feature = "neural")]
pub mod neural {
    //! Neural reranker (requires `--features neural`).
    //!
    //! Uses a BERT model via `candle-core` and `candle-transformers` for
    //! contextual reranking.  Each candidate path is scored by computing a
    //! proxy for sentence naturalness from the mean-pooled BERT hidden states.
    //! Lower scores correspond to more natural (central in embedding space)
    //! segmentations.
    //!
    //! # Model Loading
    //!
    //! Models are loaded **lazily** on the first call to [`Reranker::rerank`]
    //! and cached via [`std::sync::OnceLock`].  If the model directory is
    //! missing or malformed the reranker silently falls back to cost-based
    //! selection, so the pipeline remains fully functional without a model.
    //!
    //! # Required Files
    //!
    //! The model directory must contain:
    //! - `tokenizer.json` — HuggingFace fast tokenizer
    //! - `config.json`    — BERT model configuration
    //! - `model.safetensors` — model weights in safetensors format
    //!
    //! Compatible models: `bert-base-japanese`, `bert-tiny-japanese`,
    //! `tohoku-nlp/bert-base-japanese-v3`, etc.

    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use candle_transformers::models::bert::{BertModel, Config as BertConfig};
    use std::path::PathBuf;
    use std::sync::OnceLock;
    use tokenizers::Tokenizer;

    use super::{RerankCandidate, Reranker};

    // ── Internal lazy state ──────────────────────────────────────────────────

    /// Loaded model bundle, cached after the first successful load.
    struct ModelBundle {
        model: BertModel,
        tokenizer: Tokenizer,
    }

    // ── NeuralReranker ───────────────────────────────────────────────────────

    /// BERT-based perplexity reranker.
    ///
    /// Scores each candidate path by computing a naturalness proxy derived
    /// from the mean-pooled BERT hidden states of the morpheme surface
    /// sequence.  The candidate with the lowest score (most central embedding)
    /// is selected as the preferred segmentation.
    ///
    /// # Graceful Degradation
    ///
    /// If model files are absent or loading fails, every candidate receives
    /// `f64::MAX` and the reranker falls back to cost-based selection
    /// (identical to [`super::CostReranker`]).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "neural")]
    /// # {
    /// use mecrab::NeuralReranker;
    /// use mecrab::rerank::{Reranker, RerankCandidate};
    ///
    /// let reranker = NeuralReranker::new("/path/to/bert-model");
    /// let candidates = vec![
    ///     RerankCandidate {
    ///         surfaces: vec!["東京".to_owned(), "は".to_owned()],
    ///         pos_tags: vec!["名詞".to_owned(), "助詞".to_owned()],
    ///         cost: 100,
    ///     },
    /// ];
    /// let best = reranker.rerank(&candidates);
    /// # }
    /// ```
    pub struct NeuralReranker {
        /// Path to a HuggingFace-format model directory.
        model_path: PathBuf,
        /// Candle execution device (CPU or Metal/CUDA if available).
        device: Device,
        /// Lazily loaded and cached model bundle.
        bundle: OnceLock<Result<ModelBundle, String>>,
    }

    impl NeuralReranker {
        /// Create a new `NeuralReranker` pointing to the given model directory.
        ///
        /// Model files are **not** loaded here; loading is deferred to the
        /// first call to [`Reranker::rerank`].
        pub fn new(model_path: impl Into<PathBuf>) -> Self {
            Self {
                model_path: model_path.into(),
                device: Device::Cpu,
                bundle: OnceLock::new(),
            }
        }

        /// Override the inference device.
        ///
        /// Defaults to [`Device::Cpu`].  Pass `Device::new_metal(0)` or
        /// `Device::new_cuda(0)` for GPU acceleration when available.
        #[must_use]
        pub fn with_device(mut self, device: Device) -> Self {
            self.device = device;
            self
        }

        /// Return the configured model path.
        pub fn model_path(&self) -> &std::path::Path {
            &self.model_path
        }

        // ── Private helpers ──────────────────────────────────────────────────

        /// Load the model bundle on first access; return a reference to the
        /// cached result thereafter.
        fn get_bundle(&self) -> Option<&ModelBundle> {
            let result = self
                .bundle
                .get_or_init(|| load_bundle(&self.model_path, &self.device));
            result.as_ref().ok()
        }

        /// Compute a naturalness proxy score for a surface sequence.
        ///
        /// Joins surfaces with spaces, encodes with the BERT tokenizer, runs a
        /// forward pass and returns the L2-squared norm of the mean-pooled
        /// sequence output.  Lower values indicate more central (natural)
        /// representations.
        ///
        /// Returns `f64::MAX` on any error so the candidate is ranked last.
        fn score_candidate(&self, surfaces: &[String]) -> f64 {
            let bundle = match self.get_bundle() {
                Some(b) => b,
                None => return f64::INFINITY,
            };

            let text = surfaces.join(" ");
            let encoding = match bundle.tokenizer.encode(text.as_str(), false) {
                Ok(e) => e,
                Err(_) => return f64::INFINITY,
            };

            let ids: Vec<u32> = encoding.get_ids().to_vec();
            if ids.is_empty() {
                return f64::INFINITY;
            }

            // Build input_ids tensor: shape [1, seq_len]
            let input_ids =
                match Tensor::new(ids.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(_) => return f64::INFINITY,
                };

            // Build token_type_ids (all zeros, single segment)
            let zeros: Vec<u32> = vec![0u32; ids.len()];
            let token_type_ids =
                match Tensor::new(zeros.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(_) => return f64::INFINITY,
                };

            // Forward pass: [1, seq_len, hidden_size]
            let sequence_output = match bundle.model.forward(&input_ids, &token_type_ids, None) {
                Ok(out) => out,
                Err(_) => return f64::INFINITY,
            };

            // Mean-pool over sequence dimension → [1, hidden_size]
            // Then compute sum of squared values as a scalar naturalness proxy.
            // More "central" (natural) sequences yield smaller norms.
            compute_naturalness_score(&sequence_output).unwrap_or(f64::INFINITY)
        }
    }

    // ── Free functions ───────────────────────────────────────────────────────

    /// Load a [`ModelBundle`] from the given directory.
    fn load_bundle(model_path: &std::path::Path, device: &Device) -> Result<ModelBundle, String> {
        // 1. Tokenizer
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("tokenizer load failed: {e}"))?;

        // 2. BERT config
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("config read failed: {e}"))?;
        let config: BertConfig =
            serde_json::from_str(&config_str).map_err(|e| format!("config parse failed: {e}"))?;

        // 3. Weights (safetensors, memory-mapped for minimal startup cost)
        let weights_path = model_path.join("model.safetensors");
        // SAFETY: The file is treated as read-only; no aliased mutable
        // references can exist.  This is the standard usage pattern for
        // mmap-based weight loading in candle.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)
                .map_err(|e| format!("weight load failed: {e}"))?
        };

        // 4. Build model
        let model = BertModel::load(vb, &config).map_err(|e| format!("model build failed: {e}"))?;

        Ok(ModelBundle { model, tokenizer })
    }

    /// Compute the L2-squared norm of the mean-pooled sequence output.
    ///
    /// Returns `f64::INFINITY` on any tensor error to signal that the score
    /// is unavailable (triggers cost-based fallback in the reranker).
    fn compute_naturalness_score(sequence_output: &Tensor) -> Option<f64> {
        // Mean pool over token dimension (dim 1): [1, seq_len, hidden] → [1, hidden]
        let mean_pooled = sequence_output.mean(1).ok()?;
        // Element-wise square then sum-all: scalar proportional to ||emb||²
        let score_tensor = mean_pooled.sqr().ok()?.sum_all().ok()?;
        let score_f32: f32 = score_tensor.to_scalar().ok()?;
        Some(score_f32 as f64)
    }

    // ── Trait implementations ────────────────────────────────────────────────

    impl std::fmt::Debug for NeuralReranker {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("NeuralReranker")
                .field("model_path", &self.model_path)
                .field("device", &self.device)
                .finish_non_exhaustive()
        }
    }

    impl Reranker for NeuralReranker {
        fn rerank(&self, candidates: &[RerankCandidate]) -> usize {
            if candidates.is_empty() {
                return 0;
            }

            let scores: Vec<f64> = candidates
                .iter()
                .map(|c| self.score_candidate(&c.surfaces))
                .collect();

            // All f64::MAX → model unavailable, fall back to cost-based
            let all_worst = scores
                .iter()
                .all(|s| s.is_infinite() && s.is_sign_positive());
            if all_worst {
                return candidates
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, c)| c.cost)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }

            scores
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }

        fn name(&self) -> &'static str {
            "NeuralReranker"
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates() -> Vec<RerankCandidate> {
        vec![
            RerankCandidate {
                surfaces: vec!["東京".to_owned(), "は".to_owned()],
                pos_tags: vec!["名詞".to_owned(), "助詞".to_owned()],
                cost: 100,
            },
            RerankCandidate {
                surfaces: vec!["東".to_owned(), "京は".to_owned()],
                pos_tags: vec!["名詞".to_owned(), "名詞".to_owned()],
                cost: 200,
            },
            RerankCandidate {
                surfaces: vec!["東京は".to_owned()],
                pos_tags: vec!["名詞".to_owned()],
                cost: 50,
            },
        ]
    }

    // ── NullReranker ──────────────────────────────────────────────────────────

    #[test]
    fn test_null_reranker_always_returns_0() {
        let reranker = NullReranker;
        let candidates = make_candidates();
        // Regardless of which candidate has the minimum cost, NullReranker picks 0.
        assert_eq!(reranker.rerank(&candidates), 0);
    }

    #[test]
    fn test_null_reranker_empty_candidates() {
        let reranker = NullReranker;
        assert_eq!(reranker.rerank(&[]), 0);
    }

    #[test]
    fn test_null_reranker_single_candidate() {
        let reranker = NullReranker;
        let single = vec![RerankCandidate {
            surfaces: vec!["東京は".to_owned()],
            pos_tags: vec!["名詞".to_owned()],
            cost: 42,
        }];
        assert_eq!(reranker.rerank(&single), 0);
    }

    #[test]
    fn test_null_reranker_name() {
        assert_eq!(NullReranker.name(), "NullReranker");
    }

    // ── CostReranker ─────────────────────────────────────────────────────────

    #[test]
    fn test_cost_reranker_picks_minimum() {
        let reranker = CostReranker;
        let candidates = make_candidates();
        // Index 2 has cost=50, which is the minimum.
        assert_eq!(reranker.rerank(&candidates), 2);
    }

    #[test]
    fn test_cost_reranker_empty_candidates() {
        let reranker = CostReranker;
        assert_eq!(reranker.rerank(&[]), 0);
    }

    #[test]
    fn test_cost_reranker_single_candidate() {
        let reranker = CostReranker;
        let single = vec![RerankCandidate {
            surfaces: vec!["東京は".to_owned()],
            pos_tags: vec!["名詞".to_owned()],
            cost: 999,
        }];
        assert_eq!(reranker.rerank(&single), 0);
    }

    #[test]
    fn test_cost_reranker_name() {
        assert_eq!(CostReranker.name(), "CostReranker");
    }

    #[test]
    fn test_cost_reranker_tied_costs_picks_first() {
        // When two candidates have equal cost, `min_by_key` returns the first
        // one found (Rust's iterator guarantees stable leftmost-first for equal
        // keys in `min_by_key`).
        let reranker = CostReranker;
        let tied = vec![
            RerankCandidate {
                surfaces: vec!["A".to_owned()],
                pos_tags: vec!["名詞".to_owned()],
                cost: 10,
            },
            RerankCandidate {
                surfaces: vec!["B".to_owned()],
                pos_tags: vec!["名詞".to_owned()],
                cost: 10,
            },
        ];
        assert_eq!(reranker.rerank(&tied), 0);
    }

    // ── Trait object dispatch ────────────────────────────────────────────────

    #[test]
    fn test_reranker_trait_object_null() {
        let reranker: &dyn Reranker = &NullReranker;
        assert_eq!(reranker.rerank(&make_candidates()), 0);
        assert_eq!(reranker.name(), "NullReranker");
    }

    #[test]
    fn test_reranker_trait_object_cost() {
        let reranker: &dyn Reranker = &CostReranker;
        assert_eq!(reranker.rerank(&make_candidates()), 2);
        assert_eq!(reranker.name(), "CostReranker");
    }

    // ── Box<dyn Reranker> (Send + Sync) ──────────────────────────────────────

    #[test]
    fn test_boxed_reranker_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NullReranker>();
        assert_send_sync::<CostReranker>();
    }

    // ── RerankCandidate ──────────────────────────────────────────────────────

    #[test]
    fn test_rerank_candidate_equality() {
        let a = RerankCandidate {
            surfaces: vec!["東京".to_owned()],
            pos_tags: vec!["名詞".to_owned()],
            cost: 100,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── NeuralReranker (feature-gated) ───────────────────────────────────────

    #[cfg(feature = "neural")]
    mod neural_tests {
        use super::super::Reranker;
        use super::super::neural::NeuralReranker;
        use super::make_candidates;

        #[test]
        fn test_neural_reranker_fallback_to_cost_when_no_model() {
            // When the model directory is absent, NeuralReranker falls back to
            // cost-based selection (all scores are f64::MAX, so the minimum
            // cost candidate wins — index 2 with cost=50).
            let reranker = NeuralReranker::new("/tmp/nonexistent-bert-model-xyz");
            let candidates = make_candidates();
            // Index 2 has minimum cost=50.
            assert_eq!(reranker.rerank(&candidates), 2);
        }

        #[test]
        fn test_neural_reranker_name() {
            let reranker = NeuralReranker::new("/tmp/dummy-model");
            assert_eq!(reranker.name(), "NeuralReranker");
        }

        #[test]
        fn test_neural_reranker_model_path() {
            let reranker = NeuralReranker::new("/tmp/my-bert-model");
            assert_eq!(
                reranker.model_path(),
                std::path::Path::new("/tmp/my-bert-model")
            );
        }

        #[test]
        fn test_neural_reranker_empty_candidates() {
            let reranker = NeuralReranker::new("/tmp/dummy-model");
            assert_eq!(reranker.rerank(&[]), 0);
        }

        #[test]
        fn test_neural_reranker_single_candidate_no_model() {
            // Single candidate always wins (index 0) regardless of model state.
            let reranker = NeuralReranker::new("/tmp/nonexistent-bert-model-xyz");
            let single = vec![super::super::RerankCandidate {
                surfaces: vec!["東京は".to_owned()],
                pos_tags: vec!["名詞".to_owned()],
                cost: 42,
            }];
            assert_eq!(reranker.rerank(&single), 0);
        }

        #[test]
        fn test_neural_reranker_is_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<NeuralReranker>();
        }

        #[test]
        fn test_neural_reranker_debug() {
            let reranker = NeuralReranker::new("/tmp/my-bert-model");
            let dbg = format!("{reranker:?}");
            assert!(dbg.contains("NeuralReranker"));
            assert!(dbg.contains("my-bert-model"));
        }
    }
}
