# KizaMe (刻め!)

CLI for MeCrab morphological analyzer.

**MeCab → KizaMe (刻め = "Chop up!")**

## Installation

```bash
# Default (lightweight)
cargo install kizame

# With Wikidata builder
cargo install kizame --features builder
```

## Commands

### Parse (Default)
```bash
# Basic parsing
echo "すもももももももものうち" | kizame
kizame -d /var/lib/mecab/dic/ipadic-utf8 parse

# Wakati (space-separated)
echo "日本語" | kizame -w

# JSON output
echo "東京都" | kizame -O json

# With IPA pronunciation (requires | cat for terminal display)
echo "こんにちは" | kizame parse --with-ipa | cat

# With word embeddings
echo "東京に行く" | kizame parse --with-vector -v vectors.bin | cat

# With both IPA and vectors
echo "私は学生です" | kizame parse --with-ipa --with-vector -v vectors.bin | cat
```

### Dict
```bash
kizame dict init          # Find IPADIC
kizame dict info          # Show stats
kizame dict dump -d /path # Inspect
kizame dict dump -d /path --vocab > vocab.txt  # Extract vocabulary
```

### Vectors
```bash
# Train Word2Vec embeddings (Pure Rust!)
kizame vectors train \
  -i corpus_ids.txt \
  -o vectors.bin \
  -f mcv1 \
  --max-word-id 392126 \
  --size 100 \
  --window 5 \
  --negative 5 \
  --epochs 3 \
  --threads 8

# Convert word2vec text format to MCV1
kizame vectors convert \
  -i word2vec.txt \
  -o vectors.bin \
  -f word2vec-text \
  -v vocab.txt

# Show vector file info
kizame vectors info -v vectors.bin
```

### Explore (Interactive TUI Debugger)

```bash
# Launch interactive lattice debugger
kizame explore "東京に行く"

# With custom dictionary
kizame explore -d /path/to/dict "テキスト"

# With semantic pool
kizame explore -s /path/to/semantic.bin "東京都"
```

**Screenshot:**

![TUI Debugger](../docs/tui.png)

**Features:**
- Interactive lattice visualization with cost breakdown
- Vim-style navigation (h/j/k/l, arrows)
- Best path highlighting
- Node-by-node cost inspection
- Connection cost visualization
- Press `?` for help, `q` to quit

### Build (--features builder)
```bash
kizame build \
  --source ipadic.csv \
  --wikidata latest-all.json.gz \
  --output ./semantic-dic
```

## Options

| Flag | Description |
|------|-------------|
| `-d, --dicdir` | Dictionary path |
| `-O, --output-format` | default/wakati/dump/json/jsonld/turtle/ntriples/nquads |
| `-w, --wakati` | Space-separated output |
| `-n, --nbest` | N-best output count |
| `--with-ipa` | Include IPA pronunciation (requires \| cat) |
| `--with-vector` | Include word embeddings |
| `-v, --vector-pool` | Path to vector file (MCV1 format) |
| `--with-semantic` | Include Wikidata URIs (semantic formats) |

## Output Formats

| Format | Description |
|--------|-------------|
| `default` | MeCab-compatible output |
| `wakati` | Space-separated surface forms |
| `dump` | Full token details |
| `json` | JSON array output |
| `jsonld` | JSON-LD with semantic URIs |
| `turtle` | Turtle (TTL) RDF format |
| `ntriples` | N-Triples RDF format |
| `nquads` | N-Quads RDF format |

## Examples

```bash
# Basic parsing
echo "東京都庁で会議" | kizame

# JSON-LD with semantic information
echo "東京都" | kizame -O jsonld --with-semantic

# Turtle (TTL) RDF output
echo "東京に行く" | kizame -O turtle

# N-Triples RDF output
echo "東京に行く" | kizame -O ntriples

# N-Quads RDF output
echo "東京に行く" | kizame -O nquads

# N-best paths
echo "すもも" | kizame -n 5

# Streaming large files
cat large_corpus.txt | kizame -w > tokenized.txt
```

## License

MIT OR Apache-2.0
