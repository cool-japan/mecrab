# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- **Real-runtime SIMD Viterbi** (`viterbi/simd/`): `is_x86_feature_detected!("avx2"|"sse4.1")` runtime dispatch so AVX2/SSE4.1 actually activate on stock `cargo build --release` without requiring `RUSTFLAGS=-C target-cpu=native`. New `batch_min_argmin_i64` kernel (4×i64 AVX2, 2×i64 SSE4.1, NEON, scalar) replaces the scalar accumulate+argmin in `scan_predecessors` — the actual Viterbi hot loop is now vectorized. 9 bit-identical correctness tests vs. scalar reference.
- **IPA/X-SAMPA long-vowel collapsing** (`phonetic/transducer.rs`): post-pass collapses おう/おお→`oː`, うう→`ɯː`, えい/ええ→`eː`, ああ→`aː`, いい→`iː`; X-SAMPA uses `:` notation. `to_ipa("とうきょう")` now correctly returns `toːkʲoː`.
- **`ContextBased` disambiguation now active** (`semantic/disambiguation.rs`): reads `context_words` (token-overlap scoring +0.1/match) and `topic_hints` (URI-substring boost ×1.3) — was previously a no-op.
- **Wikidata POS→entity-type filter fully implemented** (`mecrab-builder/src/wikidata/`): P31 "instance of" Q-IDs now stored in `WikidataIndex` during dump parsing; processor's `Some(non-empty)` Q-ID filter now performs the type intersection — the carefully-curated POS-to-Q-ID tables are now live.
- **CoNLL-U dependency heuristic uses case particles** (`api/format.rs`): が→`nsubj`, を→`obj`, に/へ/で→`obl`, の→`nmod` (genitive attaches to the following noun, not root) — was position-vs-root.
- **Benchmark suite now runs without IPADIC** (`mecrab-bench`): `viterbi.rs` no longer panics; `parsing.rs`, `latticeprob.rs`, `dictionary.rs` fall back to `build_synthetic_dictionary()` — the forward-backward / `parse_with_probs` benchmarks now run in CI for the first time.
- **487 tests** (was 454): +33 new — SIMD correctness (×9), IPA long-vowel (×5), ContextBased (×4), SemanticPool (×3), CoNLL-U case particles (×1), vector quantization round-trips (×4), word2vec trainer CPU+GPU-fallback (×2+1), E2E add_word/overlay/probs/BPE/reranker (×5).

### Fixed

- `SemanticPool::write_to` now rejects prefixes >63 bytes with a clear `Error` instead of silently truncating (potential data loss on long IRI prefixes).

### Removed

- Dead `FeatureTable::from_bytes` stub (zero callers; real path is `SysDic::get_feature`).
- Dead `CachedMatrix` struct and `cached_matrix.rs` (zero constructors in production code).
- Dead `CharDefCached` struct and impl (zero constructors in production code).

---

- **GPU-accelerated word2vec** (`--features gpu`): wgpu-backed skip-gram training with a hand-written WGSL compute shader; graceful CPU Hogwild! fallback when no adapter is found. `kizame vectors train --gpu` flag added. `TrainingConfig::use_gpu` field + `Word2VecBuilder::use_gpu()` API.
- **Real NFKC normalization**: replaced hand-rolled "simplified NFKC-like" normalizer with the `unicode-normalization` crate (㌔→キロ, ①→1, combining marks, all NFC/NFD/NFKD forms).
- **F16/I8 vector quantization**: `VectorStore::get_dequantized()` decodes half-precision and int8 vectors; `quantize_f16()` and `quantize_i8()` writers produce valid MCV1 binary files; `mean_pooling` now works for all data types.
- **`MeCrab::from_dictionary` and `MeCrab::from_bytes`**: construct a fully functional analyzer from an in-memory `Dictionary` or four raw byte slices — no filesystem access required.
- **Synthetic dictionary generator** (`mecrab-builder`): `build_synthetic_dictionary()` produces a complete, small-but-real Japanese MeCab dictionary (sys.dic + matrix.bin + char.bin + unk.dic) in memory, curated to correctly disambiguate the classic sentence "すもももももももものうち".
- **First end-to-end pipeline test** (`mecrab-builder/tests/end_to_end.rs`): 7 integration tests covering full parse, wakati, feature carry-through, n-best, unknown-word handling, CoNLL-U format, and empty-input safety — all passing without a real IPADIC install.
- `mecrab-builder`: `build_matrix_bytes`, `build_char_bytes`, `build_sysdic_bytes` / `build_unkdic_bytes` in-memory binary writers.
- `kizame`: `--gpu` flag on `vectors train` (advisory when compiled without `--features gpu`).
- Fixed: `vocab_size - 1` integer underflow in `mecrab-word2vec` on empty corpus.
- Fixed: `ViterbiNode` → `PathNode` rename propagated to `wasm/format.rs`.
- Fixed: `x86_impl` / `x86_gather` doc comments added; `scalar_min_forward` / `scalar_min_from` / `find_index_of` `#[allow(dead_code)]` on baseline x86-64 (only reachable from AVX2/SSE4.1 code paths).

- Phase 3.2: `OutputFormat::LatticeProb` — JSON output with marginal probabilities from forward-backward algorithm (enables subword regularization for LLM pre-training)
- Phase 3.2: `OutputFormat::BpeCompatible` — SentencePiece-style output with ▁ word-initial markers
- `MeCrab::parse_with_probs()` — returns analysis result plus `LatticeProbTable` with per-morpheme marginal probabilities
- `ViterbiSolver::forward_backward()` — full forward-backward algorithm with numerically stable log-sum-exp
- `LatticeProbTable` and `NodeMarginal` types in `mecrab::viterbi::analysis`
- kizame CLI: `-O latticeprob` and `-O bpecompatible` output format flags
- WASM: `MeCrabWasm::parse_bpe_compatible()` and `parse_lattice_prob()` methods
- Python bindings: `MeCrab.parse_bpe()` and `MeCrab.parse_with_probs()` methods
- Lattice DOT visualization: probability overlay support via `DotBuilder::with_probs()`
- mecrab-bench: reranker overhead benchmarks (`benches/rerank.rs`)
- mecrab-bench: LatticeProb vs Viterbi overhead benchmarks (`benches/latticeprob.rs`)
- mecrab-word2vec: end-to-end integration tests for train/save/load pipeline
- `DictionaryProvider` trait: comprehensive test coverage for IPADIC/UniDic/NEologd/AutoDetect
- WASM: 27 integration tests covering error handling, Unicode edge cases, API surface
- Bug fix: `vocab_size - 1` integer underflow in `mecrab-word2vec` when vocabulary is empty

## [0.2.0] - 2026-03-29

### Added

- Hybrid SoA (Struct of Arrays) ViterbiTable for cache-efficient Viterbi algorithm
- Multi-platform SIMD acceleration: ARM NEON (aarch64), x86_64 AVX2/SSE4.1, WebAssembly SIMD128
- DAT prefetch hints for aarch64 and x86_64 platforms
- CachedMatrix: 256-slot direct-mapped connection matrix row cache
- CharDefCached: 256-slot character info cache
- Python bindings (pyo3 0.28): full sequence protocol, PyAnalysisResult, PyAnalysisResultIterator
- Batch API: `parse_batch_with_progress`, `parse_iter`, `parse_nbest_batch`, `wakati_batch_with_progress`, `wakati_iter`
- mecrab-builder: Wikidata streaming processor with rayon parallel parsing
- mecrab-builder: Wikipedia abstract integration
- mecrab-builder: DBpedia NTriples processor
- mecrab-builder: Online entity resolver (Wikidata API)
- mecrab-builder: Custom ontology import (CSV/JSON/RDF/XML)
- mecrab-builder: Direct sys.dic binary generation (MeCab-compatible format)
- mecrab-builder: Confidence calibration and disambiguation improvements
- mecrab-bench: Benchmark regression tracking
- GitHub Actions CI (ubuntu + macOS, nextest, clippy, llvm-cov, Codecov)
- GitHub Actions release workflow

### Fixed

- EOS token bug (Issue #1): backward pass now correctly identifies EOS entries

## [0.1.0] - 2026-01-20

Initial release on PyPI.
