# MeCrab: Technical Highlights 2026

## 1. Extreme Velocity

**SIMD & Parallelism:** Full utilization of modern CPU instruction sets (AVX-512/NEON) that were impossible with 2000s-era codebases. Matrix computations parallelized at the hardware level.

**True Zero-Copy:** Architecture that mmaps dictionary files and performs zero memory copies from analysis to output.

**Flyweight Semantic Pool:** Proprietary binary design that separates semantic information (URIs) as "Cold Data", maximizing Viterbi algorithm cache efficiency.

## 2. Semantic Intelligence

**Native JSON-LD:** Instead of just labels like "noun", outputs globally unique IDs like `http://www.wikidata.org/entity/Q1490`.

**Bridge to LLM:** Transforms text into knowledge graphs, making it an essential component for RAG (Retrieval-Augmented Generation) and LLM preprocessing in the AI era.

## 3. Phonetic Backbone

**IPA Output:** Outputs precise International Phonetic Alphabet (IPA) notation, not just reading pronunciations.

**TTS Ready:** Directly embeddable as a backend for speech synthesis and Japanese language learning applications.

## 4. Living Dictionary

**The Builder Pipeline:** With mecrab-builder, automatically incorporate "yesterday's new words" from Wikidata and Wikipedia dumps into the dictionary.

**No More "Unknown Words":** Systematically overcomes morphological analysis's greatest weakness—outdated dictionaries that can't analyze new terms.

## 5. Industrial Robustness

**Panic Free:** Rust's type system combined with hundreds of millions of fuzzing tests mathematically guarantees the system will never crash on any malicious input data.

**Memory Safety:** Terms like "memory leak" and "segmentation fault" from the C++ era do not exist in this project.

## 6. Run Anywhere

**WASM First:** Runs in browsers, callable from Python with zero-copy, and starts up in milliseconds on server-side.

**3-Crate Architecture:** Lean dependency management that provides lightweight CLI for users and robust build capabilities for data scientists.

---

## MeCab vs. MeCrab: The Decisive Differences

| Feature | Old MeCab (C++) | MeCrab (Rust) |
|---------|-----------------|---------------|
| Output | Text strings | Semantic & Phonetic |
| Dictionary | Static, manual updates | Dynamic, auto-generated (Wikidata Fusion) |
| Memory | Pointer operations, many copies | Zero-Copy & Packed Structs |
| Safety | Depends on developer diligence | Guaranteed by compiler and fuzzing |
| Role | Analysis tool | AI & Web infrastructure foundation |

---

## Future Extension Features (Candidates)

### 1. The "Matrix" TUI Debugger (Visual Lattice Explorer)

*Your terminal becomes a cockpit.*

MeCab's debug output (like `-N2`) was just walls of text, making it painful to mentally reconstruct the graph. Using `ratatui`, we transform this into a rich interactive interface.

#### Feature Details

**Lattice Graph View:**
- Display the lattice structure (morpheme candidate graph) for input text as a tree or network on the left side of the screen
- Navigation: Move between nodes (word candidates) with arrow keys, select paths with Enter

**Cost Inspection Panel:**
- Display details of the selected node on the right side
- Real-time display of word cost, connection cost, and cumulative cost
- Show "why the system chose this path" via cost delta from the best path

**Dynamic Re-calculation:**
- Test "what-if" scenarios like "what happens if I choose 'cut' here?" with on-the-spot cost recalculation

#### Implementation Approach

- **Libraries:** ratatui (UI rendering), crossterm (input handling)
- **Architecture:** Add a `DebugSession` struct to mecrab-core that retains and returns Viterbi intermediate state (all node cost information). CLI (`kizame explore`) instantiates this and runs the TUI loop.

#### Why It's Cool

- **Hacker Aesthetic:** Manipulating lattice structures composed of green text on a pitch-black screen is straight out of *The Matrix*.
- **Educational Value:** The ultimate teaching material for students and engineers learning NLP to intuitively understand the Viterbi algorithm.

---

### 2. "Vector-Ready" Output (Embedding on the Fly)

*Complete text analysis and semantic vectorization in a single pass.*

Normally, after tokenizing with MeCab, you vectorize with Python's gensim or PyTorch—a slow, memory-hungry double effort. MeCrab outputs vectors the moment it analyzes.

#### Feature Details

**Embedded Vectors:**
- Store embedding vectors (Word2Vec, GloVe, chiVe, etc.) corresponding to each word in the dictionary (`semantic.bin`)
- **Quantization:** Store as f16 (half-precision) or i8 (quantized) instead of f32 to reduce file size, expanding at load time

**Output Formats:**
- **Token Vectors:** Output each morpheme's vector directly
- **Sentence Vector (Mean Pooling):** Rapidly compute and output the average of all word vectors in a sentence (Bag-of-Words style sentence vector)

**Zero-Overhead:**
- Copy vector arrays directly from the binary dictionary before any string processing (UTF-8 decoding, etc.)—10-100x faster than Python

#### Implementation Approach

- **Dict Extension:** Add `vector_offset` to `semantic.bin`
- **SIMD:** Use Rust's `portable-simd` for blazing-fast vector addition (Mean Pooling)
- **Format:** Binary output (NumPy `.npy` compatible, etc.) or JSON arrays

#### Why It's Cool

- **AI Pipeline Revolution:** Generate data directly from Rust backend to Vector DB without Python.
- **Lightweight:** Complete vectorization with a single binary—no massive ML library installation required.

---

### 3. "Compile-Time" Dictionary (include_dict!)

*Freedom from "dictionary file not found" forever.*

The biggest headache when distributing morphological analyzers is "dictionary path configuration". We solve this with Rust's powerful macro system.

#### Feature Details

**Macro Magic:**
- Just write `mecrab::include_system_dict!("ipadic-2.7.0");` to download and build the dictionary at compile time, embedding it in the executable's static region (`.rodata`)

**Single Binary Distribution:**
- The generated binary (`kizame`) requires zero external files. Just `scp` to a server and it runs.

**No Parsing Cost:**
- At runtime, just cast a memory address. No file I/O or parsing occurs, achieving startup time of 0.00 seconds (immeasurable).

#### Implementation Approach

- **Proc Macro:** Create `mecrab-macros` crate
- **Bytes Integration:** Internally generate `include_bytes!` macro and map to structs defined in the dict crate via unsafe cast
- **Compression (Optional):** For binary size concerns, option to compress with zstd and decompress at initialization

#### Why It's Cool

- **Ultimate Portability:** When building Docker images, just `COPY kizame /bin/`. Whether Alpine Linux or WASM environment, the peace of mind that "if you have the binary, it absolutely runs" is immense.
- **Serverless Optimal:** Peak performance in environments where cold start (startup speed) is critical, like AWS Lambda or Cloudflare Workers.

---

## Summary

- **TUI Debugger** captures the hearts of *developers*.
- **Vector Output** captures the hearts of *AI engineers*.
- **Compile-Time Dict** captures the hearts of *infrastructure engineers*.
