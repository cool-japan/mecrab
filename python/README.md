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

### `mecrab.version() -> str`

Get MeCrab version string.

**Returns:** Version string (e.g., "0.1.0")

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

## Limitations & Future Work

### Current Limitations

1. **Vector training required separately**
   - Use `mecrab-word2vec` CLI tool to train embeddings
   - Vectors must be indexed by dictionary word_ids
   - Future: Direct Python API for these features

2. **No JSON-LD output from Python API**
   - Use CLI for JSON-LD with semantic URIs
   - Future: Native JSON output methods

### Planned Enhancements

```python
# Future API (roadmap)
m = mecrab.MeCrab(
    dicdir="/var/lib/mecab/dic/ipadic-utf8",
    with_ipa=True,                    # Enable IPA pronunciation
    vector_path="vectors.bin",        # Load word embeddings
    semantic_pool="semantic.bin"      # Load semantic URIs
)

# Parse to dictionary (Pythonic API)
result = m.parse_to_dict("東京に行く")
# => [
#   {
#     'surface': '東京',
#     'pos': '名詞',
#     'pos_detail': ['固有名詞', '地域', '一般'],
#     'reading': 'トウキョウ',
#     'pronunciation': 'トーキョー',
#     'ipa': '/toːkʲoː/',
#     'embedding': array([0.1, -0.2, 0.3, ...]),
#     'semantic_uri': 'http://www.wikidata.org/entity/Q1490',
#     'wcost': 3003
#   },
#   ...
# ]

# Semantic similarity
similar = m.most_similar("東京", top_n=10)
# => [('京都', 0.85), ('大阪', 0.82), ...]

# JSON-LD export
json_ld = m.parse_to_jsonld("東京に行く")
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
