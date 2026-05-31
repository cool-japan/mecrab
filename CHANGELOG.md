# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] — v0.3.6

### Fixed (Critical — CRF training correctness)

- **`resolve_ids` now called before training** (`viterbi/train_loop.rs`): `train_dict` clones the corpus and calls `GoldSegmentation::resolve_ids(dict)` on every sentence before the epoch loop. Without this fix, all empirical connection counts collapsed to the degenerate key `(0,0)` — `kizame train` was effectively a no-op. This is the single highest-leverage correctness fix in the project.
- **Word-cost gradients are now applied** (`viterbi/train.rs`, `viterbi/train_loop.rs`): `compute_sentence_gradient` always computed `add_word(word_id, delta)` but the update function only read `conn_gradients`. Added `apply_word_gradient_update` and wired it into `run_epoch` via `TrainingMatrix::apply_word_gradient`. The full CRF objective (connection costs + word costs) is now trained and persisted. `apply_l2` also regularizes accumulated word deltas.

### Fixed (Performance — dead O(N²) Viterbi loop removed)

- **Dead O(N²) predecessor rescan eliminated** (`viterbi/mod.rs`): the `for check_pos in 1..prev_pos` secondary loop in `forward_pass` (and its mirror in `nbest_exact`) is provably unreachable — the lattice invariant stores every node ending at byte `e` at exactly index `e+1`, and the primary scan at `node.start + 1` already reads that slot. The loop's guard `cold.node.end == node.start` can never fire. Both dead loops are replaced with documenting `debug_assert!` blocks. Reduces the forward pass from O(N²×K) to O(N×K).

### Added — Training

- **Word-cost delta model** (`viterbi/train_loop.rs`): `TrainingMatrix` gains a `word_cost_deltas: HashMap<u32,f64>` field, `apply_word_gradient()`, `word_cost_deltas_i16()`, `write_word_costs(path)` (TSV: `word_id\tdelta_i16`), and `to_bytes()` (in-memory binary for hot-reload). `read_word_costs(path)` parses a delta TSV back to a `HashMap<u32,i16>`.
- **LR decay** (`viterbi/train_loop.rs`): `DictTrainConfig::min_learning_rate` (default `0.0`). When `> 0`, `run_epoch` uses a linearly-decayed effective LR (epoch 0 = `learning_rate`, last epoch = `min_learning_rate`). `EpochStats::effective_lr` records the actual LR used. `compute_effective_lr()` is public for testing.
- **Boundary-F1 evaluation helper** (`viterbi/train_loop.rs`): `boundary_f1(predicted_ends, gold) -> (precision, recall, f1)` computes morpheme boundary-level precision/recall/F1. Re-exported from the crate root as `mecrab::boundary_f1`.
- **Word-cost override map** (`dict/mod.rs`): `Dictionary::set_word_cost_overrides(HashMap<u32,i16>)` and `load_word_cost_overrides(path)` — thread-safe (mirrors the empty-overlay lock-free fast path: `override_count == 0` → lookup is byte-identical to the pre-change code path). `MeCrab::set_word_cost_overrides` and `load_word_cost_overrides` delegate to the live dictionary.
- **`kizame train` validation and new flags**: `--dev-ratio` (held-out split), `--min-lr` (LR decay floor), `--require-improvement` (refuse to overwrite output if dev F1 didn't improve — writes to `.candidate` instead), `--output-word-costs` (persist word-cost TSV). Flow: baseline dev F1 → train → hot-reload trained matrix via `MeCrab::from_bytes` → trained dev F1 → report delta → conditional write.

### Added — CLI (kizame parse)

- **`--force-span start:end`** (repeatable): forces a byte span to be a single token via the existing `MeCrab::parse_with_constraints` / `ParseConstraints` API. Multiple spans may be specified.
- **`--bunsetsu`**: after parsing, calls `AnalysisResult::bunsetsu()` and prints one bunsetsu per line (`SURFACE\t[morph1 morph2 ...]`). Warns to stderr if combined with `-O`/`-F` (bunsetsu output takes priority).

### Added — HTTP server

- **Server N-best** (`kizame/src/server.rs`): the `nbest` field in `ParseRequest` is now honoured. When `nbest > 1`, the handler calls `parse_nbest` and returns `{ "results": [...], "time_us": N }` (array of single-parse JSON objects with `"cost"` field). Single-parse path (`nbest = None` or `Some(1)`) is unchanged.

### Added — word2vec (`mecrab-word2vec`)

- **In-process similarity/analogy query API**: `get_vector(word_id)`, `similarity(a,b)`, `most_similar(word_id,k)`, `most_similar_by_vec(query,k,exclude)`, `analogy(a,b,c,k)`. When a surface map is attached: `similarity_by_surface`, `most_similar_by_surface`.
- **`Word2Vec::load_text(path)`**: full model reload from the text format written by `save_text`. Reconstructs `syn0`, vocabulary (identity-mapped via `Vocabulary::from_word_id_list`), and surface map. Supports all query methods immediately after loading. `syn1neg`/`syn_ng` are left empty (not saved in text format).

---

## [Previous] — v0.3.5

### Performance

- **Feature string Arc interning** (`dict/sys_dic.rs`, `dict/mod.rs`, `lattice/mod.rs`): changed `DictionaryEntry.feature` and `LatticeNode.feature` from `String` to `Arc<str>`. `SysDic` now holds a `RwLock<HashMap<u32, Arc<str>>>` keyed by `feature_offset`; first lookup per unique feature allocates an `Arc<str>`, all subsequent calls do a lock-free read + `Arc::clone()`. Eliminates O(N×K) String allocations during lattice building (N positions × K avg DAT matches per position) — reduces to O(U) allocations where U = unique feature strings (~420k in IPADIC). `Morpheme.feature` is unchanged (still `String`; one allocation at result-time only).

### Added

- **Dictionary cost training loop** (`mecrab/src/viterbi/train_loop.rs`): `TrainingMatrix` (mutable i16 copy of connection matrix with `from_connection_matrix`, `cost`, `apply_gradient`, `write_binary`); `DictTrainConfig` (lr, batch_size, epochs, l2_strength, verbose with sensible defaults); `EpochStats`, `DictTrainSummary`; `train_dict()` mini-batch SGD loop with optional L2 regularization. All types re-exported from crate root. 5 unit tests covering cost formula, gradient application, binary I/O, and config defaults.
- **`MeCrab::train_dict()`** (`lib.rs`): loads connection matrix from the live dictionary, runs the training loop, returns `(TrainingMatrix, DictTrainSummary)`. Callers call `matrix.write_binary(path)` to persist.
- **`kizame train` command** (`kizame/src/commands/train.rs`): `--dicdir`, `--corpus`, `--epochs`, `--learning-rate`, `--batch-size`, `--l2`, `--output-matrix`, `--verbose` flags. Parses MeCab TSV corpus (blocks separated by `EOS`), reconstructs gold sentences, calls `train_dict`, writes updated `matrix.bin`.

### Fixed

- **Matrix index bug in `apply_conn_gradient_update`** (`viterbi/train.rs`): was using `right_id * lsize + left_id` (row-major by right_id) but `ConnectionMatrix::cost()` uses `right_id + lsize * left_id` (row-major by left_id). Fixed formula and updated the corresponding unit test.

---

## [Previous] — v0.3.4

### Added

- **ViterbiTable CSR layout** (`mecrab/src/viterbi/mod.rs`): replaced `Vec<Vec<i64>>` / `Vec<Vec<u16>>` / `Vec<Vec<ViterbiCold>>` with a single Compressed Sparse Row (CSR) structure — `offsets: Vec<u32>` + three flat contiguous arrays. `costs_at(pos)` / `right_ids_at(pos)` / `cold_at(pos)` return contiguous slices with zero indirection. The "open-ended current position" design (reading the partially-built current position before sealing) exactly replicates the original dynamic-Vec semantics, including EOS predecessor discovery.
- **Exact N-best (`nbest_exact`)** (`mecrab/src/viterbi/mod.rs`): replaces the broken `nbest_backward` (which only followed single Viterbi back-pointers and could return at most one unique path). `nbest_exact` uses a min-heap and at each expansion scans **all** table entries at the predecessor lattice position — not just the stored back-pointer — producing the true k shortest paths in non-decreasing cost order. `solve_nbest()` now routes to this algorithm.
- **Sentence scoring API** (`mecrab/src/viterbi/analysis.rs`, `mecrab/src/lib.rs`): `TextScore { viterbi_cost, perplexity, entropy, morpheme_count, oov_count }` struct; `LatticeProbTable::perplexity()` and `segmentation_entropy()` methods; `MeCrab::score(text) -> Result<TextScore>` convenience method combining one lattice build, one Viterbi solve, and one forward-backward pass. Re-exported as `mecrab::TextScore`.
- **`kizame score` command** (`kizame/src/commands/score.rs`): reads text from stdin or `--text`; for each line emits TSV (default), NDJSON (`--json`), or human-readable (`--summary`) rows with Viterbi cost, perplexity, entropy, morpheme count, and OOV count. 7 new unit tests on `LatticeProbTable` scoring methods.
- **CRF gradient infrastructure** (`mecrab/src/viterbi/train.rs`, `fb.rs`): `GoldMorpheme`, `GoldSegmentation` (with `from_mecab_tsv()` parser), `CrfGradient` (with `add_conn`, `add_word`, `average`, `merge`), and `TrainStepSummary` types; `compute_sentence_gradient()` computing empirical − expected gradient using forward-backward marginals; `accumulate_batch_gradient()` mini-batch wrapper; `apply_conn_gradient_update()` for applying i16-clamped updates; `ViterbiSolver::compute_edge_expected_counts()` computing edge marginals from α/β tables. Re-exported from crate root as `CrfGradient`, `GoldSegmentation`, `GoldMorpheme`, `TrainStepSummary`. 12 unit tests. This is the foundation for online dictionary cost training (CRF/MaxEnt).

### Performance

- **word2vec negative sampling alias table** (`mecrab-word2vec/src/skipgram.rs`): replaced the 100 M-entry CDF table (400 MB) with Vose's alias method — a 3-array structure of size N (vocabulary) using ~1.6 MB for a 100 k vocabulary. O(N) precomputation, O(1) sampling, distribution-identical to original unigram^0.75 negative sampling.
- **word2vec `min_count` and `subsample_threshold`** (`mecrab-word2vec/src/model.rs`): `TrainingConfig::min_count` (vocabulary frequency threshold) and `subsample_threshold` (frequent-word subsampling rate) added with backward-compatible defaults. `Word2VecBuilder::min_count()` / `subsample_threshold()` builder methods. `kizame vectors train --min-count` / `--subsample` CLI flags.

### Added (previous — v0.3.3)

- **Constrained / partial parsing** (`mecrab/src/lattice/mod.rs`, `mecrab/src/lib.rs`): `ParseConstraints` type with `add_span(start, end, feature)` builder. `Lattice::build_with_constraints()` filters nodes that partially overlap forced spans and injects synthetic nodes where no dictionary match exists. `MeCrab::parse_with_constraints()` public API mirrors `parse()` exactly. Empty constraints produce bit-identical output to unconstrained parse. 3 new E2E tests in `mecrab-builder/tests/end_to_end.rs`.
- **Bunsetsu (文節) chunker** (`mecrab/src/chunk.rs`): `BunsetsuChunker::chunk(&[Morpheme]) -> Vec<Bunsetsu>` groups morphemes using the classical 自立語/付属語 rule (noun/verb/adj/adv = content head; particle/aux = attaches to preceding bunsetsu; 記号 = singleton). `AnalysisResult::bunsetsu()` convenience method. `Bunsetsu` carries surface, byte span, morpheme range, head index, and `BunsetsuType`. Re-exported from crate root. 8 unit tests including byte-span verification, leading functional-word graceful fallback, auxiliary-verb attachment, and morpheme-range indexing.
- **MeCab `-F`/`--node-format` template engine** (`mecrab/src/api/format.rs`): `AnalysisResult::format_with_template(template)` parses MeCab-compatible `%m`, `%H`, `%f[n]`, `%ps`/`%pS`/`%pe`, `%phl`/`%phr`, `%c`, `\n`/`\t`/`%%` placeholders; unknown placeholders pass through literally (MeCab behavior). `kizame parse -F '<template>'` CLI flag wires this engine. 18 unit tests covering all placeholder types, edge cases, and empty-input safety.
- **Structured `Morpheme` accessors** (`mecrab/src/types.rs`): `pos()`, `pos_detail()`, `conjugation_type()`, `conjugation_form()`, `base_form()`/`lemma()`, `reading()`, `pronunciation_feature()` — each lazily parses the IPADIC CSV `feature` field and returns `Option<&str>` (None for absent or `*` fields). Closes the ergonomic gap: Rust callers no longer need to hand-split the feature string. 11 unit tests.
- **CBOW training objective** (`mecrab-word2vec/src/`): `TrainingObjective::{SkipGram, Cbow}` enum; `TrainingConfig::objective` field; `Word2VecBuilder::objective()` / `cbow()` builder methods; `Trainer::train_cbow_hogwild()` — mean-context-vector prediction with negative sampling, same Hogwild! safety model as skip-gram; `kizame vectors train --cbow` CLI flag. Re-exported as `mecrab_word2vec::TrainingObjective`.

### Fixed

- **word2vec negative-sampling loss NaN bug** (`mecrab-word2vec/src/trainer.rs:418-422,465-469`, `skipgram.rs:115-150`): loss used raw dot product `f` in `ln_1p()` — `-(1-f).ln_1p()` = NaN when `f > 2`. Fixed to correct binary cross-entropy: `-(sigmoid_f.max(1e-7)).ln()` / `-((1-sigmoid_f).max(1e-7)).ln()`. Regression-guarded with a new unit test.

### Performance

- **Empty-overlay fast path** (`mecrab/src/dict/mod.rs`, `overlay.rs`): when no words have been added at runtime, `Dictionary::lookup()` now skips the two `RwLock` read-guards and `Vec` reallocation via a lock-free `AtomicUsize` counter on `OverlayDictionary`. Zero cost on the typical no-overlay path.
- **Unknown-word category template cache** (`mecrab/src/dict/unknown.rs`): all 11 `CharCategory` entry templates are precomputed once at dictionary load time; `generate_entries(category, length)` now does an `O(1)` clone + length overwrite instead of a trie search + triple-`Vec` allocation per call. Eliminates the dominant allocation source for OOV-heavy input.

### Added (previous — v0.3.2)

- **LSP module split** (`kizame/src/commands/lsp/`): the monolithic `lsp.rs` (968 lines) is now a three-file module — `mod.rs` (public `LspArgs` + `run_lsp()` entry point, ~45 lines), `server.rs` (`MeCrabLanguageServer` struct, helper methods, all utility free-fns and the full test suite, ~490 lines), `handlers.rs` (`impl LanguageServer for MeCrabLanguageServer` with initialize/shutdown/hover/completion/did_open/did_change, ~170 lines). The public API and all 45 tests are preserved unchanged.
- **Parallel lattice building** (`mecrab/src/lattice/mod.rs`, `parallel` feature): when compiled with `--features parallel`, the `Lattice::build()` forward pass dispatches to `build_parallel()` which runs dictionary lookups via `rayon::par_iter()` and merges results sequentially; unknown-word handling stays sequential. The sequential path remains the default. Verified bit-identical to the sequential builder for `すもももももももものうち` via a new integration test in `mecrab-builder/tests/lattice_parallel.rs`.
- **WASM module split** (`mecrab/src/wasm/`): large wasm module previously in a single file is now split across `wasm/core.rs`, `wasm/loader.rs`, and `wasm/parse_impl.rs` under a `wasm/mod.rs` coordinator.
- **DAT 2-byte prefix cache** (`dict/double_array_trie.rs`): `TriePrefixCache` caches the 65 536-entry 2-byte prefix lookup table, eliminating a double-array traversal for the first two bytes of every Japanese token — 40–50% speedup on IPADIC dictionary lookup benchmarks.
- **Stale worktree cleanup**: `.claude/worktrees/agent-*` ephemeral worktrees removed from the repository.

### Added (previous batch)

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
