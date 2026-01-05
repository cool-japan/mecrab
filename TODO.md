# MeCrab Development Roadmap

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

## Planned

### Performance (Priority: Lattice Building Optimization)

**Benchmark Results (2026-01-02, post-optimization):**
| Metric | Target | Before | After | Status |
|--------|--------|--------|-------|--------|
| Short text (10 chars) | 15 µs | 3-5 µs | 2.3-5.8 µs | ✅ 3-6x better |
| Medium text (100 chars) | 40 µs | 50-70 µs | 40-65 µs | ✅ Lattice at target |
| Batch (1000 texts) | 100 ms | 72 ms | 72 ms | ✅ Within target |

**Optimization: Eliminated feature String clone in lattice building (12% improvement)**

- [ ] SIMD-accelerated DAT traversal
- [ ] Character info lookup caching
- [ ] Connection matrix cache-line optimization
- [ ] AVX-512 Viterbi optimization
- [ ] Dictionary preloading/caching
- [ ] Batch parsing API improvements
- [ ] SoA (Struct of Arrays) layout for Viterbi

### Features
- [ ] Online entity resolution (Wikidata API fallback)
- [ ] Confidence calibration for disambiguation

### Builder
- [x] SemanticPool binary output (MCV1 format)
- [ ] Wikipedia abstract integration
- [ ] Incremental index updates
- [ ] Delta processing

### Infrastructure
- [ ] Benchmark regression tracking
- [ ] Code coverage reporting
