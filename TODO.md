# MeCrab Development Roadmap

## v0.3.0 Implemented

### Phase 3.2 - LLM-Ready Output Formats
- [x] Forward-backward algorithm (LatticeProbTable, NodeMarginal, log_sum_exp)
- [x] LatticeProb output format (JSON with marginal probabilities)
- [x] BpeCompatible output format (SentencePiece ▁-marked tokens)
- [x] parse_with_probs() API
- [x] Python bindings: parse_bpe(), parse_with_probs()
- [x] WASM bindings: parse_bpe_compatible(), parse_lattice_prob()
- [x] kizame CLI: -O latticeprob / -O bpecompatible
- [x] Lattice DOT visualization with probability overlay
- [x] LatticeProb vs Viterbi benchmark
- [x] Reranker overhead benchmarks (NullReranker/CostReranker)
- [x] word2vec end-to-end integration tests
- [x] DictionaryProvider comprehensive tests
- [x] WASM integration test expansion (27 tests)

## Completed

### Core Library (mecrab)
- [x] Double-Array Trie dictionary loading
- [x] Viterbi algorithm with path optimization
- [x] Memory-mapped dictionary files
- [x] Character category definitions (char.def)
- [x] Connection matrix (matrix.def)
- [x] Unknown word handling
- [x] Thread-safe design
- [x] Live dictionary overlay (add/remove words)

### Phase 2: Semantic Enrichment (ED-002, ED-004)
- [x] SemanticPool (5-byte compact entries)
- [x] TokenExtension for multi-candidate URIs
- [x] JSON-LD export
- [x] RDF/Turtle/N-Triples/N-Quads export
- [x] SPARQL query generation
- [x] Disambiguation strategies (HighestConfidence, PopularityPrior, ContextWindow)
- [x] Dictionary semantic loading (load_with_semantics)
- [x] Surface form → URI mapping (surface_map.json)
- [x] MeCrab parser integration with entities
- [x] CLI `--with-semantic` flag (opt-in semantic output)
- [x] Conditional entity population (lazy evaluation)

### Phase 3: Performance (ED-003)
- [x] Portable SIMD (nightly `std::simd`)
- [x] 8-way parallel batch cost addition
- [x] SIMD minimum finding with lane reduction
- [x] SIMD predecessor batch processing

### Phase 4: Advanced Features
- [x] N-best path search (A* algorithm)
- [x] Path ranking and cost analysis
- [x] Streaming text processing
- [x] Sentence boundary detection
- [x] Text normalization (NFKC, width, case)
- [x] Phonetic transduction (Kana/Romaji/X-SAMPA/IPA)
- [x] User dictionary persistence
- [x] Benchmarking utilities

### CLI (kizame)
- [x] Basic parse command
- [x] Output formats (default, wakati, dump, json)
- [x] `dict init` - find/install dictionaries
- [x] `dict dump` - inspect dictionary contents
- [x] `dict info` - show dictionary statistics
- [x] Feature-gated `build` command

### Builder (mecrab-builder)
- [x] 3-crate architecture (ED-006)
- [x] Dependency isolation
- [x] Wikidata JSON streaming parser
- [x] Surface → URI index builder

### Infrastructure
- [x] Fuzz testing (4 targets: viterbi, dictionary, parser, lattice)
- [x] Comprehensive test suite (168+ tests)

### Phase 5: Bindings
- [x] Python bindings (PyO3) packaging
- [x] WASM bindings packaging
- [x] TypeScript type declarations (.pyi and .d.ts)

### Visualization
- [x] Lattice visualization (DOT/Graphviz)
- [x] DotBuilder for configurable output
- [x] Multiple layout directions (LR, TB, RL, BT)

### CLI (kizame) - continued
- [x] `dict compile` - compile CSV to binary
- [x] N-best output mode (`-n` flag)
- [x] JSON-LD output format (`-O jsonld`)
- [x] Color terminal output (`-c` / `--no-color`)
- [x] Streaming input for large files
- [x] Progress indicators for long operations
- [x] `explore` TUI command (Matrix mode lattice debugger)
- [x] `serve` HTTP API server (POST /parse, POST /parse/batch, GET /wakati, GET /health)

### TUI Debugger (kizame explore)
- [x] Interactive lattice visualization with ratatui
- [x] Vim-style navigation (h/j/k/l, arrows)
- [x] Cost panel with Viterbi breakdown
- [x] Best path highlighting
- [x] Node-by-node cost inspection
- [x] Connection cost visualization
- [x] UTF-8 safe feature string truncation
- [x] Auto-scrolling for long lattices
- [x] Semantic pool support (`-s` flag)

**Screenshot:** [docs/tui.png](docs/tui.png)

### Benchmark Suite (mecrab-bench)
- [x] Separate subcrate for benchmarks
- [x] Short/medium/long text parsing benchmarks
- [x] Batch processing benchmarks (100/500/1000 texts)
- [x] N-best search benchmarks
- [x] Lattice vs Viterbi component breakdown
- [x] Dictionary loading benchmarks
- [x] Cache effects analysis

### Word Embeddings (mecrab/vectors)
- [x] Zero-copy vector store with mmap (docs/007.md)
- [x] Binary format (MCV1) for word embeddings
- [x] Support for f32/f16/i8 data types
- [x] Mean pooling for sentence vectors
- [x] Cosine similarity computation
- [x] Pure Rust implementation with bytemuck


## 0.2.0 Implemented

### Core Fixes
- [x] EOS token bug fix (Issue #1) — backward_pass now correctly identifies EOS entries
- [x] Hybrid SoA ViterbiTable with hot/cold separation

### SIMD Optimizations
- [x] ARM NEON (aarch64) batch cost operations
- [x] x86_64 AVX2/SSE4.1 batch cost operations
- [x] WebAssembly SIMD128 batch cost operations
- [x] DAT prefetch hints (aarch64 + x86_64)
- [x] SIMD batch connection cost lookups (viterbi/simd.rs `batch_connection_costs`)
- [x] CachedMatrix: 256-slot direct-mapped connection matrix row cache
- [x] CharDefCached: 256-slot character info lookup cache

### Dictionary
- [x] DictionaryProvider trait (IPADIC/UniDic/NEologd/Auto)
- [x] AutoDetect: samples feature count from first dict entry
- [x] Dictionary::from_bytes — load from raw bytes (WASM support)
- [x] Direct sys.dic binary generation (MeCab-compatible format)

### Python Bindings (pyo3 0.28)
- [x] Full sequence protocol (PyAnalysisResult, PyAnalysisResultIterator)
- [x] GIL-release batch processing (py.detach())
- [x] python/ submodule structure (analysis/parser/helpers)

### WASM
- [x] load_dictionary from packed blob (36-byte header format)
- [x] parse/parse_wakati full pipeline
- [x] 13 WASM integration tests
- [x] wasm/pack_dict.js Node.js helper

### Builder (mecrab-builder)
- [x] Wikipedia abstract integration
- [x] DBpedia NTriples processor
- [x] Online entity resolver (Wikidata API)
- [x] Custom ontology import (CSV/JSON/RDF/XML)
- [x] Confidence calibration and disambiguation
- [x] wikidata/ submodule (parser/index/mod)
- [x] CSV export with URIs

### Batch/Stream API
- [x] parse_batch_with_progress (parallel, AtomicUsize)
- [x] parse_iter, wakati_iter (lazy iterators)
- [x] parse_nbest_batch
- [x] api/ submodule (batch/iter/format)

### Reranking
- [x] Reranker trait + NullReranker + CostReranker
- [x] parse_nbest_with_reranker()
- [x] NeuralReranker stub (--features neural)

### Corpus/Word2Vec
- [x] SurfaceVocab + CorpusStats
- [x] MeCrab::tokenize_for_corpus / tokenize_corpus_lines
- [x] kizame vectors tokenize command

### LSP (--features lsp)
- [x] kizame lsp command (tower-lsp)
- [x] hover (morpheme info), completion, diagnostics (unknown words)

### CLI (kizame)
- [x] commands/ submodule refactoring
- [x] --dict-format flag (ipadic/unidic/neologd/auto)
- [x] serve output format support (json/wakati/dump)

### Benchmarks
- [x] mecrab-bench/benches/simd.rs — SIMD vs scalar benchmark
- [x] BenchBaseline regression tracking

### Infrastructure
- [x] CHANGELOG.md
- [x] Makefile with clippy-all/build-simd/build-lsp/check-wasm targets
- [x] CI: simd feature build, lsp feature build, wasm32 cross-check

## Planned

### Performance (Priority: Lattice Building Optimization)

**Benchmark Results (2026-01-02, post-optimization):**
| Metric | Target | Before | After | Status |
|--------|--------|--------|-------|--------|
| Short text (10 chars) | 15 µs | 3-5 µs | 2.3-5.8 µs | ✅ 3-6x better |
| Medium text (100 chars) | 40 µs | 50-70 µs | 40-65 µs | ✅ Lattice at target |
| Batch (1000 texts) | 100 ms | 72 ms | 72 ms | ✅ Within target |

**Optimization: Eliminated feature String clone in lattice building (12% improvement)**

- [x] SIMD-accelerated DAT traversal (prefetch_for_key + warm_cache in double_array_trie.rs)
- [x] Character info lookup caching
- [x] Connection matrix cache-line optimization
- [x] AVX-512 Viterbi optimization (implemented as AVX2/SSE4.1 — the production-relevant x86_64 targets)
- [x] Dictionary preloading/caching
- [x] Batch parsing API improvements
- [x] SoA (Struct of Arrays) layout for Viterbi (ViterbiTable hybrid SoA + CachedMatrix)

### Features
- [x] Online entity resolution (Wikidata API fallback)
- [x] Confidence calibration for disambiguation (calibrated_confidence + recalibrate())

### Builder
- [x] SemanticPool binary output (MCV1 format)
- [x] Parallel rayon processing (10k-line chunks, parallel JSON parse)
- [x] Entity type filtering via P31 claims (entity_type_filter in BuildConfig)
- [x] POS-based URI filtering (人名/地名/組織/固有名詞 IPADIC mapping)
- [x] Max-confidence duplicate detection (HashMap<uri, f32> dedup)
- [x] Wikipedia abstract integration
- [x] Incremental index updates
- [x] Delta processing
- [x] Custom ontology import (CSV / JSON / RDF/XML → WikidataIndex merge, Phase 0)

### Infrastructure
- [x] Benchmark regression tracking
- [x] Code coverage reporting (cargo-llvm-cov + Codecov via GitHub Actions CI)
- [x] GitHub Actions CI (ubuntu + macos matrix, clippy -D warnings, nextest)
- [x] GitHub Actions release workflow (auto release notes on v* tags, draft)
- [x] Direct sys.dic generation — DA-trie binary builder implemented in mecrab-builder
