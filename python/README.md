# MeCrab Python Bindings

**High-performance morphological analyzer for Python**

Pure Rust implementation with Python bindings via PyO3.

## Features

- **Fast**: Written in Rust, ~10x faster than pure Python alternatives
- **Thread-safe**: Concurrent access with zero-copy where possible
- **Batch processing**: Parallel processing for multiple texts (requires `parallel` feature)
- **Live dictionary updates**: Add custom words at runtime
- **MeCab compatible**: Works with IPADIC/UniDic dictionaries
- **Word embeddings**: Zero-copy vector integration with cosine similarity
- **IPA pronunciation**: International Phonetic Alphabet output with one-shot API
- **Semantic URIs**: Wikidata/DBpedia linking (via CLI)

## Installation

### From Source

Requires Rust toolchain (1.75+):

```bash
# Clone repository
git clone https://github.com/kitasan/mecrab.git
cd mecrab

# Build and install with maturin
pip install maturin
maturin develop --features python,parallel --release

# Or for development
maturin develop --features python,parallel
```

### From PyPI (Coming Soon)

```bash
pip install mecrab
```

### System Requirements

**IPADIC Dictionary** (required):

```bash
# Ubuntu/Debian
sudo apt install mecab-ipadic-utf8

# Fedora/RHEL
sudo dnf install mecab-ipadic

# macOS
brew install mecab-ipadic
```

Common installation paths:
- Ubuntu/Debian: `/var/lib/mecab/dic/ipadic-utf8`
- Fedora/RHEL: `/usr/lib/mecab/dic/ipadic-utf8`
- macOS: `/usr/local/lib/mecab/dic/ipadic-utf8`

## Quick Start

### Basic Parsing

```python
import mecrab

# Initialize with default dictionary
m = mecrab.MeCrab()

# Parse text
result = m.parse("すもももももももものうち")
print(result)
# =>
# すもも  名詞,一般,*,*,*,*,すもも,スモモ,スモモ
# も      助詞,係助詞,*,*,*,*,も,モ,モ
# もも    名詞,一般,*,*,*,*,もも,モモ,モモ
# も      助詞,係助詞,*,*,*,*,も,モ,モ
# もも    名詞,一般,*,*,*,*,もも,モモ,モモ
# の      助詞,連体化,*,*,*,*,の,ノ,ノ
# うち    名詞,非自立,副詞可能,*,*,*,うち,ウチ,ウチ
# EOS
```

### Wakati (Space-separated)

```python
words = m.wakati("私は学生です")
print(words)
# => "私 は 学生 です"

# As list
print(words.split())
# => ['私', 'は', '学生', 'です']
```

### Structured Output

```python
morphemes = m.parse_to_list("東京に行く")

for surface, feature in morphemes:
    parts = feature.split(',')
    pos = parts[0]  # Part-of-speech
    reading = parts[7] if len(parts) > 7 else ""
    print(f"{surface:8s} {pos:6s} {reading}")
# =>
# 東京     名詞   トウキョウ
# に       助詞   ニ
# 行く     動詞   イク
```

### Batch Processing

Process multiple texts in parallel:

```python
texts = [
    "今日は良い天気です",
    "明日は雨が降るでしょう",
    "週末は買い物に行きます"
]

# Batch parse (parallel)
results = m.parse_batch(texts)
for result in results:
    print(result)

# Batch wakati
words_list = m.wakati_batch(texts)
for words in words_list:
    print(words)
```

### Custom Dictionary (Overlay)

Add custom words at runtime:

```python
# Add new word
m.add_word(
    "ChatGPT",                    # surface form
    "チャットジーピーティー",      # reading (katakana)
    "チャットジーピーティー",      # pronunciation (katakana)
    5000                          # word cost (lower = more preferred)
)

result = m.parse("ChatGPTは便利です")
print(result)
# => Now "ChatGPT" is recognized as a single word

# Remove word
m.remove_word("ChatGPT")

# Check overlay size
print(f"Custom words: {m.overlay_size()}")
```

## Advanced Usage

### Specify Dictionary Path

```python
m = mecrab.MeCrab(dicdir="/var/lib/mecab/dic/ipadic-utf8")
```

### Extract Part-of-Speech

```python
def extract_nouns(text):
    morphemes = m.parse_to_list(text)
    return [surface for surface, feature in morphemes
            if feature.startswith('名詞')]

nouns = extract_nouns("東京の美しい公園で散歩しました")
print(nouns)
# => ['東京', '公園', '散歩']
```

### Word Frequency Counter

```python
from collections import Counter

corpus = """
東京は日本の首都です。
東京には多くの人が住んでいます。
""".strip()

all_words = []
for line in corpus.split('\n'):
    if line:
        words = m.wakati(line).split()
        all_words.extend(words)

freq = Counter(all_words)
for word, count in freq.most_common(10):
    print(f"{word:10s} {count:3d}")
```

## CLI Tools (KizaMe)

For advanced features, use the `kizame` CLI tool:

### Install CLI

```bash
cargo install --path kizame
# Or with builder features
cargo install --path kizame --features builder
```

### IPA Pronunciation

```bash
echo "東京に行く" | kizame parse --with-ipa | cat
# =>
# 東京    名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー
#   IPA: /toːkʲoː/
# に      助詞,格助詞,一般,*,*,*,に,ニ,ニ
#   IPA: /ɲi/
# 行く    動詞,自立,*,*,五段・カ行促音便,基本形,行く,イク,イク
#   IPA: /ikɯ/
# EOS
```

> **Note:** IPA output requires `| cat` for proper terminal display due to Unicode handling.

### Word Embeddings

#### Option 1: Use Pre-trained Vectors (Recommended for Quick Start)

Download pre-trained Japanese Wikipedia vectors from HuggingFace:

```bash
# Install huggingface_hub
pip install huggingface-hub

# Download vectors (MCV1 format, ~65MB)
huggingface-cli download KitaSan/mecrab-jawiki-word2vec vectors.bin --local-dir .

# Or use Python API
python3 << 'EOF'
from huggingface_hub import hf_hub_download
vectors_path = hf_hub_download(
    repo_id="KitaSan/mecrab-jawiki-word2vec",
    filename="vectors.bin",
    repo_type="dataset"
)
print(f"Downloaded to: {vectors_path}")
EOF
```

**Then use in Python:**

```python
import mecrab

# Use downloaded vectors
m = mecrab.MeCrab(vector_path="vectors.bin")

# Compute similarity
sim = m.similarity("東京", "京都")
print(f"Similarity: {sim:.3f}")  # => High similarity

# With IPA pronunciation
m = mecrab.MeCrab(with_ipa=True, vector_path="vectors.bin")
morphemes = m.parse_to_dict("東京に行く")
for morph in morphemes:
    print(f"{morph['surface']}: IPA=/{morph.get('ipa')}/, embedding={len(morph.get('embedding', []))}-dim")
```

**Dataset Details:**
- **Source**: Japanese Wikipedia (全文)
- **Vocabulary**: 163,922 words (IPADIC)
- **Vector Size**: 100 dimensions
- **Training**: Hogwild! algorithm (83% parallel efficiency on 6 cores)
- **Format**: MCV1 (memory-mapped, instant loading)
- **HuggingFace**: https://huggingface.co/datasets/KitaSan/mecrab-jawiki-word2vec

#### Option 2: Train Custom Vectors

For domain-specific corpora, train your own vectors:

```bash
# Extract vocabulary
kizame dict dump -d /var/lib/mecab/dic/ipadic-utf8 --vocab > vocab.txt
MAX_WORD_ID=$(tail -1 vocab.txt | cut -f1)

# Parse corpus to word_id sequences
cat corpus.txt | kizame parse --wakati-word-id > corpus_ids.txt

# Train Word2Vec (MCV1 binary format - memory-mapped, instant loading)
kizame vectors train \
  -i corpus_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id $MAX_WORD_ID \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 6

# Use embeddings
echo "東京に行く" | kizame parse --with-vector -v vectors.bin --with-ipa | cat
```

See [Word2Vec Training Guide](../kizame/WORD2VEC_TRAINING_GUIDE.md) for complete workflow.

### JSON-LD with Semantic URIs

```bash
echo "東京に行く" | kizame -O jsonld --with-semantic
# => JSON-LD output with Wikidata URIs
```

## Jupyter Notebooks

Interactive tutorials available in `examples/`:

1. **[01_basic_usage.ipynb](examples/01_basic_usage.ipynb)**
   - Basic parsing, wakati, batch processing
   - Custom dictionary management
   - Practical examples (word frequency, POS extraction)

2. **[02_word2vec_training.ipynb](examples/02_word2vec_training.ipynb)**
   - Complete Word2Vec training pipeline
   - Vocabulary extraction
   - Corpus parsing
   - Training and verification
   - Production deployment

### Run Notebooks

```bash
# Install Jupyter
pip install jupyter notebook

# Start Jupyter
cd python/examples
jupyter notebook
```

## API Reference

### `mecrab.MeCrab`

Main morphological analyzer class.

#### Constructor

```python
MeCrab(dicdir: Optional[str] = None,
       with_ipa: bool = False,
       vector_path: Optional[str] = None)
```

**Parameters:**
- `dicdir`: Dictionary directory path (optional, auto-detected if not specified)
- `with_ipa`: Enable IPA pronunciation output (default: False)
- `vector_path`: Path to word embeddings file (vectors.bin) (optional)

**Returns:** `MeCrab` instance

**Raises:** `RuntimeError` if dictionary or vectors cannot be loaded

**Examples:**
```python
# Basic usage
m = mecrab.MeCrab()

# With IPA pronunciation
m = mecrab.MeCrab(with_ipa=True)

# With word embeddings
m = mecrab.MeCrab(vector_path="vectors.bin")

# Combined IPA + vectors
m = mecrab.MeCrab(with_ipa=True, vector_path="vectors.bin")
```

#### Methods

##### `parse(text: str) -> str`

Parse text and return MeCab-compatible formatted string.

**Parameters:**
- `text`: Input text to analyze

**Returns:** Formatted analysis result

**Example:**
```python
result = m.parse("こんにちは")
```

---

##### `wakati(text: str) -> str`

Parse text and return space-separated surface forms.

**Parameters:**
- `text`: Input text to analyze

**Returns:** Space-separated words

**Example:**
```python
words = m.wakati("私は学生です")  # => "私 は 学生 です"
```

---

##### `parse_to_list(text: str) -> List[Tuple[str, str]]`

Parse text and return list of (surface, feature) tuples.

**Parameters:**
- `text`: Input text to analyze

**Returns:** List of morphemes as (surface, feature) tuples

**Example:**
```python
morphemes = m.parse_to_list("東京")
# => [('東京', '名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トーキョー')]
```

---

##### `parse_batch(texts: List[str]) -> List[str]`

Parse multiple texts in batch (parallel processing).

**Parameters:**
- `texts`: List of input texts

**Returns:** List of formatted analysis results

**Example:**
```python
results = m.parse_batch(["テスト1", "テスト2"])
```

---

##### `wakati_batch(texts: List[str]) -> List[str]`

Parse multiple texts and return wakati outputs in batch.

**Parameters:**
- `texts`: List of input texts

**Returns:** List of space-separated word strings

**Example:**
```python
words_list = m.wakati_batch(["こんにちは", "さようなら"])
```

---

##### `add_word(surface: str, reading: str, pronunciation: str, wcost: int)`

Add a custom word to the overlay dictionary.

**Parameters:**
- `surface`: Surface form (the actual text)
- `reading`: Katakana reading
- `pronunciation`: Katakana pronunciation
- `wcost`: Word cost (lower = more preferred, typical: 5000-8000)

**Example:**
```python
m.add_word("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000)
```

---

##### `remove_word(surface: str) -> bool`

Remove a word from the overlay dictionary.

**Parameters:**
- `surface`: Surface form to remove

**Returns:** `True` if word was removed, `False` if not found

**Example:**
```python
removed = m.remove_word("ChatGPT")
```

---

##### `overlay_size() -> int`

Get the number of words in the overlay dictionary.

**Returns:** Number of custom words

**Example:**
```python
count = m.overlay_size()
```

---

##### `to_ipa(text: str) -> List[str]`

**Requires:** `with_ipa=True` in constructor

Convert text to IPA pronunciation and return as list of strings (one per morpheme).

**Parameters:**
- `text`: Input text to convert

**Returns:** List of IPA pronunciation strings

**Raises:** `RuntimeError` if IPA not enabled or parsing fails

**Example:**
```python
m = mecrab.MeCrab(with_ipa=True)
ipas = m.to_ipa("東京に行く")
# => ['toːkʲoː', 'ɲi', 'ikɯ']

# Join with spaces
print(" ".join(ipas))
# => 'toːkʲoː ɲi ikɯ'
```

---

##### `to_ipa_text(text: str, separator: str = " ") -> str`

**Requires:** `with_ipa=True` in constructor

Convert text to IPA pronunciation and return as a single string.

**Parameters:**
- `text`: Input text to convert
- `separator`: Separator between morphemes (default: " ")

**Returns:** IPA pronunciation string

**Raises:** `RuntimeError` if IPA not enabled or parsing fails

**Example:**
```python
m = mecrab.MeCrab(with_ipa=True)

# Default separator (space)
ipa = m.to_ipa_text("東京に行く")
# => 'toːkʲoː ɲi ikɯ'

# Custom separator
ipa = m.to_ipa_text("東京に行く", separator="-")
# => 'toːkʲoː-ɲi-ikɯ'
```

---

##### `similarity(word1: str, word2: str) -> float`

**Requires:** `vector_path` in constructor

Compute cosine similarity between two words using their word embeddings.

**Parameters:**
- `word1`: First word
- `word2`: Second word

**Returns:** Cosine similarity in range [-1.0, 1.0]

**Raises:** `RuntimeError` if:
- Vectors not enabled
- Word not found in vocabulary
- Word has no embedding (out-of-vocabulary)

**Example:**
```python
m = mecrab.MeCrab(vector_path="vectors.bin")

# Compare similar words
sim = m.similarity("東京", "京都")
print(f"Similarity: {sim:.3f}")  # => 0.856 (high similarity)

# Compare dissimilar words
sim = m.similarity("東京", "食べる")
print(f"Similarity: {sim:.3f}")  # => 0.123 (low similarity)
```

**Note:** If a word tokenizes into multiple morphemes, uses the first morpheme's embedding.

---

##### `parse_to_morphemes(text: str) -> List[Morpheme]`

Parse text and return list of Morpheme objects with rich properties.

**Parameters:**
- `text`: Input text to analyze

**Returns:** List of Morpheme objects

**Example:**
```python
morphemes = m.parse_to_morphemes("東京に行く")
for morph in morphemes:
    if morph.is_noun():
        print(f"Noun: {morph.surface} ({morph.reading})")
```

---

##### `parse_nbest(text: str, n: int = 5) -> List[Tuple[List[Dict], int]]`

Parse text and return N-best analyses ranked by cost.

**Parameters:**
- `text`: Input text to analyze
- `n`: Number of best paths to return (default: 5)

**Returns:** List of (morphemes, cost) tuples sorted by cost

**Example:**
```python
results = m.parse_nbest("すもももももももものうち", n=3)
for morphemes, cost in results:
    print(f"Cost: {cost}")
```

---

##### `parse_json(text: str) -> str`

Parse text and return JSON string.

**Parameters:**
- `text`: Input text to analyze

**Returns:** JSON string

---

##### `parse_jsonld(text: str) -> str`

Parse text and return JSON-LD string with semantic annotations.

**Parameters:**
- `text`: Input text to analyze

**Returns:** JSON-LD string

---

##### `sentence_embedding(text: str) -> List[float]`

**Requires:** `vector_path` in constructor

Get sentence embedding via mean pooling of word vectors.

**Parameters:**
- `text`: Input text

**Returns:** List of floats (embedding vector)

**Example:**
```python
m = mecrab.MeCrab(vector_path="vectors.bin")
emb = m.sentence_embedding("東京に行く")
print(f"Dimension: {len(emb)}")
```

---

##### `parse_batch_to_morphemes(texts: List[str]) -> List[List[Morpheme]]`

Parse multiple texts and return Morpheme objects in batch.

**Parameters:**
- `texts`: List of texts to analyze

**Returns:** List of lists of Morpheme objects

---

##### `parse_batch_to_dict(texts: List[str]) -> List[List[Dict]]`

Parse multiple texts and return dictionaries in batch.

**Parameters:**
- `texts`: List of texts to analyze

**Returns:** List of lists of dictionaries

---

##### `dict_info() -> Dict`

Get dictionary information.

**Returns:** Dictionary with keys: `overlay_size`, `has_vectors`, `with_ipa`

---

##### Properties

- `has_vectors: bool` - Check if vector support is enabled
- `has_ipa: bool` - Check if IPA support is enabled

---

### `mecrab.Morpheme`

A morpheme object with rich properties and helper methods.

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `surface` | `str` | Surface form (actual text) |
| `feature` | `str` | Full feature string (comma-separated) |
| `pos` | `str` | Part-of-speech (main category) |
| `pos1`, `pos2`, `pos3` | `Optional[str]` | POS subcategories |
| `inflection` | `Optional[str]` | Conjugation type |
| `conjugation` | `Optional[str]` | Conjugation form |
| `base` | `Optional[str]` | Base form (lemma) |
| `reading` | `Optional[str]` | Reading (katakana) |
| `pronunciation` | `Optional[str]` | Pronunciation (katakana) |
| `ipa` | `Optional[str]` | IPA pronunciation (if enabled) |
| `embedding` | `Optional[List[float]]` | Word embedding (if enabled) |
| `pos_id` | `int` | Part-of-speech ID |
| `wcost` | `int` | Word cost |
| `word_id` | `int` | Word ID for vector lookup |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_noun()` | `bool` | Check if morpheme is a noun (名詞) |
| `is_verb()` | `bool` | Check if morpheme is a verb (動詞) |
| `is_adjective()` | `bool` | Check if morpheme is an adjective (形容詞) |
| `is_particle()` | `bool` | Check if morpheme is a particle (助詞) |
| `is_auxiliary()` | `bool` | Check if morpheme is an auxiliary verb (助動詞) |
| `is_symbol()` | `bool` | Check if morpheme is a symbol (記号) |
| `to_dict()` | `Dict` | Convert to dictionary |

---

### `mecrab.version() -> str`

Get MeCrab version string.

**Returns:** Version string (e.g., "0.2.0")

---

### `mecrab.default_dicdir() -> Optional[str]`

Get default dictionary path.

**Returns:** Dictionary path if found, None otherwise

---

### `mecrab.cosine_similarity(a: List[float], b: List[float]) -> float`

Compute cosine similarity between two vectors.

**Parameters:**
- `a`: First vector
- `b`: Second vector

**Returns:** Cosine similarity in range [-1.0, 1.0]

**Raises:** `ValueError` if vectors have different dimensions or are zero

## Feature Flags

Build-time features (for `maturin`):

| Feature | Description |
|---------|-------------|
| `python` | Enable Python bindings (required) |
| `parallel` | Enable parallel batch processing (rayon) |
| `json` | Enable JSON output support |

Example:
```bash
maturin develop --features python,parallel,json
```

## Performance

Benchmark comparison (10,000 sentences):

| Method | Time | Speedup |
|--------|------|---------|
| Python (single) | 2.5 sec | 1.0x |
| Python (batch, 8 cores) | 0.4 sec | **6.3x** |
| MeCab (original C++) | 0.3 sec | 8.3x |

MeCrab Python bindings with batch processing achieve near-C++ performance while maintaining Python's ease of use.

## Enhanced Features (New in v0.2.0)

### Morpheme Objects

Get rich Morpheme objects with helper methods:

```python
morphemes = m.parse_to_morphemes("東京に行く")
for morph in morphemes:
    print(f"{morph.surface}: pos={morph.pos}, reading={morph.reading}")

    # POS helper methods
    if morph.is_noun():
        print(f"  Noun: {morph.surface}")
    if morph.is_verb():
        print(f"  Verb: {morph.base}")  # lemma form

    # Convert to dict if needed
    d = morph.to_dict()
```

### N-best Analysis

Get alternative analyses ranked by cost:

```python
# Get top 5 alternative analyses
results = m.parse_nbest("すもももももももものうち", n=5)
for morphemes, cost in results:
    surfaces = [m['surface'] for m in morphemes]
    print(f"Cost: {cost:6d} | {' / '.join(surfaces)}")
```

### JSON Output

Export as JSON or JSON-LD:

```python
import json

# Simple JSON
json_str = m.parse_json("東京に行く")
data = json.loads(json_str)
print(data[0]['surface'])  # => "東京"

# JSON-LD with semantic context
jsonld_str = m.parse_jsonld("東京に行く")
print(jsonld_str)  # => {"@context": {...}, "tokens": [...]}
```

### Sentence Embeddings

Get sentence-level embeddings via mean pooling:

```python
m = mecrab.MeCrab(vector_path="vectors.bin")

# Get sentence embedding
emb = m.sentence_embedding("東京に行く")
print(f"Dimension: {len(emb)}")  # => 100 (or your vector size)

# Use for similarity between sentences
emb1 = m.sentence_embedding("東京に行く")
emb2 = m.sentence_embedding("京都に行く")
sim = mecrab.cosine_similarity(emb1, emb2)
print(f"Sentence similarity: {sim:.3f}")
```

### Batch Processing with Rich Output

Process batches and get rich output:

```python
texts = ["東京に行く", "京都で食べる", "大阪を歩く"]

# Batch with Morpheme objects
for morphemes in m.parse_batch_to_morphemes(texts):
    nouns = [m.surface for m in morphemes if m.is_noun()]
    print(f"Nouns: {nouns}")

# Batch with dicts
for dicts in m.parse_batch_to_dict(texts):
    print([d['surface'] for d in dicts])
```

### Context Manager Support

Use with statement for resource management:

```python
with mecrab.MeCrab() as m:
    result = m.parse("こんにちは")
    print(result)
# Resources automatically cleaned up
```

### Dictionary Info

Get information about loaded dictionaries:

```python
info = m.dict_info()
print(f"Overlay size: {info['overlay_size']}")
print(f"Has vectors: {info['has_vectors']}")
print(f"IPA enabled: {info['with_ipa']}")

# Property access
print(f"Has vectors: {m.has_vectors}")
print(f"Has IPA: {m.has_ipa}")
```

### Module-level Functions

```python
import mecrab

# Get version
print(mecrab.version())  # => "0.2.0"

# Find default dictionary path
dicdir = mecrab.default_dicdir()
print(f"Dictionary at: {dicdir}")

# Compute cosine similarity between vectors
sim = mecrab.cosine_similarity([1.0, 0.0], [0.707, 0.707])
print(f"Similarity: {sim:.3f}")
```

## Limitations & Future Work

### Current Limitations

1. **Vector training required separately**
   - Use `mecrab-word2vec` CLI tool to train embeddings
   - Vectors must be indexed by dictionary word_ids

2. **most_similar() and analogy() not yet fully implemented**
   - Requires vocabulary iteration
   - Use `similarity()` for pairwise comparison instead
   - Use `sentence_embedding()` with external ANN index for most_similar

### Future Enhancements

```python
# Future API (roadmap)
# Full most_similar with ANN index
similar = m.most_similar("東京", topn=10)
# => [('京都', 0.85), ('大阪', 0.82), ...]

# Word analogy
result = m.analogy("王様", "男", "女", topn=5)
# => [('女王', 0.92), ...]
```

## Troubleshooting

### Dictionary Not Found

**Error:** `RuntimeError: Failed to load dictionary`

**Solution:**

1. Install IPADIC:
```bash
sudo apt install mecab-ipadic-utf8
```

2. Specify dictionary path explicitly:
```python
m = mecrab.MeCrab(dicdir="/var/lib/mecab/dic/ipadic-utf8")
```

3. Find your IPADIC installation:
```bash
find /usr -name "sys.dic" 2>/dev/null | grep ipadic
```

### Import Error

**Error:** `ModuleNotFoundError: No module named 'mecrab'`

**Solution:**

Rebuild with maturin:
```bash
cd /path/to/mecrab
maturin develop --features python,parallel --release
```

### Slow Batch Processing

**Issue:** Batch processing not faster than single processing

**Solution:**

Make sure you built with `parallel` feature:
```bash
maturin develop --features python,parallel
```

## Examples

See `examples/` directory for complete notebooks:

- [Basic Usage](examples/01_basic_usage.ipynb)
- [Word2Vec Training](examples/02_word2vec_training.ipynb)

## Documentation

- **GitHub:** https://github.com/kitasan/mecrab
- **Rust API Docs:** https://docs.rs/mecrab
- **CLI Guide:** [KizaMe README](../kizame/README.md)
- **Word2Vec Training:** [Training Guide](../kizame/WORD2VEC_TRAINING_GUIDE.md)

## Development

### Build from Source

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone repository
git clone https://github.com/kitasan/mecrab.git
cd mecrab

# Install maturin
pip install maturin

# Build and install (development mode)
maturin develop --features python,parallel

# Run tests
cargo test --features python

# Build wheel (release)
maturin build --release --features python,parallel
```

### Run Tests

```bash
# Rust tests
cargo test --features python

# Python tests (if available)
pytest python/tests/
```

## License

Dual-licensed under:

- MIT License ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License 2.0 ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## Copyright

Copyright 2026 COOLJAPAN OU (Team KitaSan)

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs and feature requests, please use GitHub issues.
