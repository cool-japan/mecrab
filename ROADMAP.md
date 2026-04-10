# MeCrab: The Warp-Speed Morphological Analyzer for the AI Era

## Vision Statement

MeCrab aims to be the **world's fastest and most intelligent Japanese morphological analyzer** - not just a MeCab port, but a revolutionary engine built for 2026 and beyond.

```
🚀 Blazing Fast: SIMD-accelerated, 5x faster than MeCab
🧠 Smart: Optional Neural Reranking for ambiguous phrases
🔄 Live: Hot-swappable dictionaries without restart
🦀 Pure Rust: Memory safe, thread safe, panic free
```

---

## Phase 1: Extreme Performance (Q1 2026)

### 1.1 SIMD-Accelerated Viterbi

**Current State**: Hybrid SoA ViterbiTable implemented; SIMD via core::arch for NEON/AVX2/SSE4.1/WASM; batch connection cost lookups pending
**Target**: 4-8x speedup using portable SIMD

```rust
// Before: Sequential
for prev_entry in entries[prev_pos].iter() {
    let conn_cost = matrix.cost(prev.right_id, node.left_id);
    let total = prev_entry.cost + conn_cost + node.wcost;
    if total < best_cost { best_cost = total; }
}

// After: SIMD vectorized (8 candidates at once)
use std::simd::*;
let costs: i32x8 = prev_costs + conn_costs + Simd::splat(node.wcost);
let min_cost = costs.reduce_min();
```

**Implementation Tasks**:
- [x] Restructure ViterbiEntry for SIMD-friendly layout (SoA vs AoS)
- [x] Batch connection cost lookups (8/16 at a time)
- [x] Use `std::simd` for cost accumulation and min-finding
- [ ] Benchmark: target 5x improvement on long sentences (SIMD benchmark implemented in `mecrab-bench/benches/simd.rs`; real numbers pending real IPADIC install)

### 1.2 Cache-Optimized Double-Array Trie

**Current State**: Standard DAT with pointer-based access
**Target**: L3 cache-resident, prefetch-optimized

**Options**:
1. **daachorse integration** - Production-ready, Aho-Corasick + DAT hybrid
2. **Custom FST** - Using `fst` crate for minimal memory footprint
3. **LOUDS-based trie** - Succinct data structure, 2-3x smaller

```toml
# Cargo.toml additions for Phase 1
[dependencies]
daachorse = "1.1"  # Consider for multi-pattern matching
fst = "0.4"        # Memory-mapped FST
```

**Memory Layout Optimization**:
```rust
// Pack hot data together for cache locality
#[repr(C, align(64))]  // Cache line aligned
struct TrieBlock {
    bases: [i32; 16],   // 64 bytes - one cache line
    checks: [u32; 16],  // 64 bytes - one cache line
}
```

### 1.3 Parallel Batch Processing

**Implementation Tasks**:
- [x] Implement `parse_batch` with Rayon parallel iteration
- [x] Add `parse_batch_with_progress` and `parse_iter` APIs

```rust
use rayon::prelude::*;

impl MeCrab {
    /// Analyze multiple sentences in parallel
    pub fn parse_batch(&self, texts: &[&str]) -> Vec<Result<AnalysisResult>> {
        texts.par_iter()
            .map(|text| self.parse(text))
            .collect()
    }
}
```

---

## Phase 2: Dynamic Dictionary Architecture (Q2 2026)

### 2.1 Live Dictionary Overlay

**Implementation Tasks**:
- [x] Implement in-memory overlay layer for hot word additions
- [x] Add `add_word`, `remove_word`, and `hot_reload` APIs

```
┌─────────────────────────────────────────┐
│           MeCrab Dictionary Stack        │
├─────────────────────────────────────────┤
│  Layer 3: In-Memory Overlay (hot words)  │  ← API updates (no restart)
├─────────────────────────────────────────┤
│  Layer 2: User Dictionary (mmap)         │  ← Custom vocabulary
├─────────────────────────────────────────┤
│  Layer 1: System Dictionary (mmap)       │  ← IPADIC/UniDic/NEologd
└─────────────────────────────────────────┘
```

**API Design**:
```rust
impl MeCrab {
    /// Add a word at runtime (hot path)
    pub fn add_word(&self, surface: &str, feature: &str, cost: i16) -> Result<()>;

    /// Remove a word at runtime
    pub fn remove_word(&self, surface: &str) -> Result<()>;

    /// Reload dictionary without dropping connections
    pub async fn hot_reload(&self, path: &Path) -> Result<()>;
}
```

### 2.2 Universal Dictionary Format

**Current State**: Implemented

**Implementation Tasks**:
- [x] Define `DictionaryProvider` trait abstraction
- [x] Implement `IpadicProvider`, `UnidicProvider`, `NeologdProvider`

```rust
pub trait DictionaryProvider: Send + Sync {
    fn lookup(&self, key: &str) -> Vec<DictionaryEntry>;
    fn connection_cost(&self, right_id: u16, left_id: u16) -> i16;
    fn char_info(&self, c: char) -> CharInfo;
}

// Implementations for different formats
pub struct IpadicProvider { /* ... */ }
pub struct UnidicProvider { /* ... */ }
pub struct NeologdProvider { /* ... */ }
```

---

## Phase 3: AI Synergy (Q3 2026)

### 3.1 Neural Reranking (Optional)

**Current State**: Real BERT-based `NeuralReranker` implemented and wired via `candle-core` / `candle-transformers` behind `--features neural`. Model loading is lazy (`OnceLock`), gracefully falls back to cost-based selection when model files are absent.

```
Input: "すもももももももものうち"
       ↓
┌──────────────────┐
│  MeCrab Core     │ → Top-N lattice paths (fast, statistical)
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Neural Reranker │ → Best path selection (accurate, contextual)
│  (BERT via       │   candle-core, lazy-loaded, zero-panic
│   candle-core)   │
└──────────────────┘
         ↓
Output: Optimal segmentation
```

**Implementation Tasks**:
- [x] Define `Reranker` trait (`mecrab::rerank::Reranker`) — `Send + Sync`
- [x] Implement `NullReranker` (zero-overhead default, always picks index 0)
- [x] Implement `CostReranker` (statistical baseline, picks min-cost path)
- [x] Add `NeuralReranker` stub behind `--features neural` (Phase 3 scaffold)
- [x] Add `MeCrab::parse_nbest_with_reranker` API method
- [x] Wire `candle-core` / `candle-transformers` for actual BERT scoring
- [x] Load `bert-base-japanese` / `bert-tiny-japanese` weights lazily (`OnceLock`)
- [x] Perplexity scoring over candidate surface sequences (mean-pooled L2 norm proxy)
- [x] Reranker overhead benchmarks implemented (NullReranker vs CostReranker, N=5/10/20) in mecrab-bench

**Production API**:
```rust
// Zero-overhead: picks Viterbi-optimal (index 0)
let result = mecrab.parse_nbest_with_reranker("text", 5, &NullReranker)?;

// Statistical: re-sorts by cost (same result, tests pipeline)
let result = mecrab.parse_nbest_with_reranker("text", 5, &CostReranker)?;

// Neural (--features neural): real BERT perplexity scoring via candle-core
#[cfg(feature = "neural")]
let result = mecrab.parse_nbest_with_reranker("text", 5, &NeuralReranker::new("./model"))?;

// With explicit device (Metal/CUDA)
#[cfg(feature = "neural")]
let result = mecrab.parse_nbest_with_reranker(
    "text", 5,
    &NeuralReranker::new("./model").with_device(candle_core::Device::new_metal(0)?),
)?;
```

### 3.2 LLM-Ready Output Formats

```rust
pub enum OutputFormat {
    MeCab,           // Traditional format
    Wakati,          // Space-separated
    Json,            // Structured JSON
    LatticeProb,     // Lattice with probabilities (for Subword Regularization)
    BpeCompatible,   // Compatible with BPE tokenizers
}
```

### Phase 3.2 Implemented (LLM-Ready Output Formats)
- [x] `OutputFormat::LatticeProb` — marginal probabilities via forward-backward algorithm
- [x] `OutputFormat::BpeCompatible` — SentencePiece-style ▁-marked tokens
- [x] `MeCrab::parse_with_probs()` — combined Viterbi + forward-backward API
- [x] `ViterbiSolver::forward_backward()` — full probabilistic algorithm (T=500 Boltzmann)
- [x] Python bindings for new formats (`parse_bpe`, `parse_with_probs`)
- [x] WASM bindings for new formats (`parse_bpe_compatible`, `parse_lattice_prob`)
- [x] CLI integration: `kizame parse -O latticeprob` / `-O bpecompatible`
- [x] Lattice DOT visualization with probability overlay
- [x] LatticeProb vs Viterbi overhead benchmarks (`benches/latticeprob.rs`)

---

## Phase 4: Developer Experience (Q4 2026)

### 4.1 WebAssembly (WASM) Support

**Implementation Tasks**:
- [x] Create WASM binding stubs (`MeCrabWasm` struct, `wasm_bindgen` exports)
- [x] WASM integration tests expanded (27 tests: error handling, Unicode edge cases, API surface, binary format)

```rust
// wasm/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MeCrabWasm {
    inner: MeCrab,
}

#[wasm_bindgen]
impl MeCrabWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(dict_bytes: &[u8]) -> Result<MeCrabWasm, JsValue>;

    pub fn parse(&self, text: &str) -> String;
    pub fn wakati(&self, text: &str) -> String;
}
```

**Target Platforms**:
- Browser (wasm32-unknown-unknown)
- Cloudflare Workers (wasm32-wasi)
- Deno/Node.js

### 4.2 Python Bindings (PyO3)

**Implementation Tasks**:
- [x] Implement PyO3 bindings for `Tagger`, `parse`, `parse_batch`
- [x] Publish `mecrab` package to PyPI
- [x] Add `add_word` API for live dictionary updates from Python

```python
# pip install mecrab
import mecrab

# Simple usage
tagger = mecrab.Tagger()
result = tagger.parse("東京は日本の首都です")

# Batch processing (uses Rayon internally)
results = tagger.parse_batch(["文1", "文2", "文3"])

# Live dictionary updates
tagger.add_word("ChatGPT", "名詞,固有名詞,*,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー")
```

### 4.3 Language Server Protocol (LSP)

**Current State**: Implemented

**Implementation Tasks**:
- [x] Implement `MeCrabLanguageServer` with `tower-lsp`
- [x] Add completion and diagnostic handlers

```rust
// mecrab-lsp/src/main.rs
use tower_lsp::{LspService, Server};

struct MeCrabLanguageServer {
    mecrab: Arc<MeCrab>,
}

impl LanguageServer for MeCrabLanguageServer {
    async fn completion(&self, params: CompletionParams) -> Result<CompletionResponse>;
    async fn diagnostic(&self, params: DocumentDiagnosticParams) -> Result<DiagnosticResponse>;
}
```

---

## Benchmarking Targets

| Metric | MeCab | MeCrab v1.0 | MeCrab v2.0 (SIMD) |
|--------|-------|-------------|---------------------|
| Short text (10 chars) | 50 μs | 40 μs | 15 μs |
| Medium text (100 chars) | 200 μs | 150 μs | 40 μs |
| Long text (1000 chars) | 2 ms | 1.5 ms | 300 μs |
| Batch (1000 texts) | 2 s | 1.5 s | 100 ms (parallel) |

---

## File Structure (Future)

```
mecrab/
├── crates/
│   ├── mecrab-core/       # Core library
│   ├── mecrab-dict/       # Dictionary handling
│   ├── mecrab-simd/       # SIMD optimizations
│   ├── mecrab-neural/     # Neural reranking (optional)
│   ├── mecrab-wasm/       # WebAssembly bindings
│   ├── mecrab-python/     # Python bindings (PyO3)
│   └── mecrab-lsp/        # Language Server
├── dictionaries/
│   ├── ipadic/
│   ├── unidic/
│   └── neologd/
└── benches/
    ├── viterbi_bench.rs
    └── dictionary_bench.rs
```

---

## License

Apache 2.0 / MIT dual license

---

*MeCrab: Where Speed Meets Intelligence*

Copyright 2026 COOLJAPAN OU (Team KitaSan)
