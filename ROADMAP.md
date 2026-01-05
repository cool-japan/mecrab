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

**Current State**: Sequential cost computation in forward pass
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
- [ ] Restructure ViterbiEntry for SIMD-friendly layout (SoA vs AoS)
- [ ] Batch connection cost lookups (8/16 at a time)
- [ ] Use `std::simd` for cost accumulation and min-finding
- [ ] Benchmark: target 5x improvement on long sentences

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

```
Input: "すもももももももものうち"
       ↓
┌──────────────────┐
│  MeCrab Core     │ → Top-N lattice paths (fast, statistical)
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Neural Reranker │ → Best path selection (accurate, contextual)
│  (BERT-tiny)     │
└──────────────────┘
         ↓
Output: Optimal segmentation
```

**Implementation with Candle**:
```rust
#[cfg(feature = "neural")]
pub struct NeuralReranker {
    model: candle_transformers::bert::BertModel,
    tokenizer: tokenizers::Tokenizer,
}

impl NeuralReranker {
    pub fn rerank(&self, candidates: &[Vec<PathNode>]) -> Vec<PathNode> {
        // Score each candidate path using BERT
        // Return the highest-scoring path
    }
}
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

---

## Phase 4: Developer Experience (Q4 2026)

### 4.1 WebAssembly (WASM) Support

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
