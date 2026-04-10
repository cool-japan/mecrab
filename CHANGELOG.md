# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

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
