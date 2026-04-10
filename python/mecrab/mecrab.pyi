"""Type stubs for mecrab native module.

Copyright 2026 COOLJAPAN OU (Team KitaSan)
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

__version__: str
__author__: str


class Morpheme:
    """A single morpheme from morphological analysis.

    Provides structured access to morphological information with
    helper methods for common POS checks.
    """

    # Basic attributes (all read-only)
    surface: str
    """Surface form (the actual text)"""

    feature: str
    """Full feature string (comma-separated)"""

    pos: str
    """Part-of-speech (main category, e.g., '名詞', '動詞')"""

    pos1: Optional[str]
    """POS subcategory 1"""

    pos2: Optional[str]
    """POS subcategory 2"""

    pos3: Optional[str]
    """POS subcategory 3"""

    inflection: Optional[str]
    """Conjugation type (e.g., '五段・カ行イ音便')"""

    conjugation: Optional[str]
    """Conjugation form (e.g., '基本形', '連用形')"""

    base: Optional[str]
    """Base form / lemma"""

    reading: Optional[str]
    """Reading in katakana"""

    pronunciation: Optional[str]
    """Pronunciation in katakana"""

    ipa: Optional[str]
    """IPA pronunciation (if enabled)"""

    embedding: Optional[List[float]]
    """Word embedding vector (if enabled)"""

    pos_id: int
    """Part-of-speech ID"""

    wcost: int
    """Word cost"""

    word_id: int
    """Word ID for vector lookup"""

    # Byte position attributes
    @property
    def start_byte(self) -> int:
        """Byte offset where this morpheme starts in the input text."""
        ...

    @property
    def end_byte(self) -> int:
        """Byte offset (exclusive) where this morpheme ends in the input text."""
        ...

    # Computed properties
    @property
    def has_embedding(self) -> bool:
        """Check if morpheme has an embedding vector."""
        ...

    @property
    def has_ipa(self) -> bool:
        """Check if morpheme has IPA pronunciation."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension (0 if no embedding)."""
        ...

    # Dunder methods
    def __repr__(self) -> str:
        """Return Morpheme(surface=..., pos=..., reading=...) representation."""
        ...

    def __str__(self) -> str:
        """Return MeCab-format: surface\\tfeature."""
        ...

    # Conversion
    def to_dict(self) -> Dict[str, Any]:
        """Convert morpheme to a Python dictionary with all available fields."""
        ...

    def to_json(self) -> str:
        """Serialize morpheme to a JSON string."""
        ...

    # POS helper methods
    def is_noun(self) -> bool:
        """Check if morpheme is a noun (名詞)."""
        ...

    def is_verb(self) -> bool:
        """Check if morpheme is a verb (動詞)."""
        ...

    def is_adjective(self) -> bool:
        """Check if morpheme is an adjective (形容詞)."""
        ...

    def is_particle(self) -> bool:
        """Check if morpheme is a particle (助詞)."""
        ...

    def is_auxiliary(self) -> bool:
        """Check if morpheme is an auxiliary verb (助動詞)."""
        ...

    def is_symbol(self) -> bool:
        """Check if morpheme is a symbol/punctuation (記号)."""
        ...


class AnalysisResult:
    """Result of morphological analysis.

    Iterable, indexable container of Morpheme objects produced by
    :meth:`MeCrab.parse`.

    Example::

        result = m.parse("東京は日本の首都です")

        # Iterate
        for morpheme in result:
            print(morpheme.surface, morpheme.pos)

        # Length
        print(len(result))

        # Index (positive and negative)
        print(result[0].surface)
        print(result[-1].surface)

        # Convenience helpers
        print(result.surfaces())
        print(result.readings())
        print(result.pos_tags())

        # Serialisation
        data = result.to_list()   # List[Dict]
        json = result.to_json()   # str
    """

    text: str
    """Original input text."""

    morphemes: List[Morpheme]
    """All morphemes in order."""

    def __repr__(self) -> str: ...
    def __str__(self) -> str:
        """MeCab-format multi-line string ending with EOS."""
        ...

    def __len__(self) -> int:
        """Number of morphemes."""
        ...

    def __iter__(self) -> Iterator[Morpheme]:
        """Iterate over morphemes."""
        ...

    def __getitem__(self, idx: int) -> Morpheme:
        """Index into morphemes; supports negative indices.

        Raises:
            IndexError: If index is out of range.
        """
        ...

    def to_list(self) -> List[Dict[str, Any]]:
        """Return list of dicts, one per morpheme."""
        ...

    def to_json(self) -> str:
        """Serialize analysis to a JSON string with ``text`` and ``morphemes`` keys."""
        ...

    def surfaces(self) -> List[str]:
        """Return list of surface forms (the tokenised words)."""
        ...

    def readings(self) -> List[str]:
        """Return list of readings in katakana; empty string when not available."""
        ...

    def pos_tags(self) -> List[str]:
        """Return list of part-of-speech tags."""
        ...

    def spans(self) -> List[Tuple[int, int]]:
        """Return (start_byte, end_byte) pairs for all non-EOS morphemes."""
        ...

    def noun_phrases(self) -> List[Tuple[str, int, int]]:
        """Extract compound noun phrases.

        Returns list of (surface, start_byte, end_byte) tuples.
        Consecutive 名詞 morphemes are merged into compounds.
        """
        ...

    def named_entities(self) -> List[Tuple[str, str, int, int]]:
        """Extract named entities (固有名詞).

        Returns list of (surface, entity_type, start_byte, end_byte) tuples.
        entity_type is the IPADIC sub-category (地域, 人名, 組織, 一般, etc.)
        """
        ...

    def verb_chunks(self) -> List[Tuple[str, int, int]]:
        """Extract verb chunks (動詞 + following 助動詞).

        Returns list of (surface, start_byte, end_byte) tuples.
        """
        ...


class AnalysisResultIterator:
    """Internal iterator returned by :meth:`AnalysisResult.__iter__`."""

    def __iter__(self) -> "AnalysisResultIterator": ...
    def __next__(self) -> Morpheme: ...
    def __len__(self) -> int:
        """Remaining items."""
        ...


class AnalysisIterator:
    """Legacy streaming iterator (kept for backwards compatibility)."""

    def __iter__(self) -> "AnalysisIterator": ...
    def __next__(self) -> Morpheme: ...
    def __len__(self) -> int: ...


class NullReranker:
    """Always selects the Viterbi-optimal path (index 0). Zero overhead."""

    def __init__(self) -> None: ...
    def name(self) -> str:
        """Return 'NullReranker'."""
        ...

    def rerank(self, costs: List[int]) -> int:
        """Always return 0 regardless of the cost list."""
        ...

    def __repr__(self) -> str: ...


class CostReranker:
    """Selects the candidate with the lowest total Viterbi cost."""

    def __init__(self) -> None: ...
    def name(self) -> str:
        """Return 'CostReranker'."""
        ...

    def rerank(self, costs: List[int]) -> int:
        """Return the index of the minimum-cost candidate (0 when empty)."""
        ...

    def __repr__(self) -> str: ...


class SurfaceVocab:
    """Maps surface forms to stable integer IDs for corpus building.

    IDs are assigned in insertion order starting at 0.  Thread-safe.

    Example::

        vocab = SurfaceVocab()
        id0 = vocab.get_or_insert("東京")   # => 0
        id1 = vocab.get_or_insert("大阪")   # => 1
        id0_again = vocab.get_or_insert("東京")  # => 0
        print(len(vocab))  # => 2
    """

    def __init__(self) -> None: ...

    def get_or_insert(self, surface: str) -> int:
        """Get an existing ID or assign the next available one.

        Args:
            surface: Surface form to look up or insert.

        Returns:
            Integer ID for this surface.

        Raises:
            RuntimeError: If the internal lock is poisoned.
        """
        ...

    def get(self, surface: str) -> Optional[int]:
        """Look up the ID for surface without inserting.

        Args:
            surface: Surface form to look up.

        Returns:
            Integer ID, or None if not in the vocabulary.

        Raises:
            RuntimeError: If the internal lock is poisoned.
        """
        ...

    def __len__(self) -> int:
        """Number of distinct surface forms in the vocabulary."""
        ...

    def __repr__(self) -> str: ...


class MeCrab:
    """High-performance morphological analyzer compatible with MeCab.

    Features:
    - Fast morphological analysis with Viterbi algorithm
    - IPA (International Phonetic Alphabet) pronunciation
    - Word embeddings with cosine similarity
    - Live dictionary updates (hot-swappable)
    - Batch processing with parallel support
    - N-best path analysis
    - JSON/JSON-LD output
    - Context manager support
    """

    def __init__(
        self,
        dicdir: Optional[str] = None,
        with_ipa: bool = False,
        vector_path: Optional[str] = None,
    ) -> None:
        """Create a new MeCrab instance.

        Args:
            dicdir: Optional path to dictionary directory (auto-detected if not specified)
            with_ipa: Enable IPA pronunciation output (default: False)
            vector_path: Path to word embeddings file (vectors.bin) (optional)

        Raises:
            RuntimeError: If dictionary or vectors cannot be loaded

        Examples:
            >>> m = MeCrab()
            >>> m = MeCrab(with_ipa=True)
            >>> m = MeCrab(vector_path="vectors.bin")
            >>> m = MeCrab(with_ipa=True, vector_path="vectors.bin")
        """
        ...

    # Properties
    @property
    def has_vectors(self) -> bool:
        """Check if this instance has vector support enabled."""
        ...

    @property
    def has_ipa(self) -> bool:
        """Check if this instance has IPA support enabled."""
        ...

    # Core parsing
    def parse(self, text: str) -> AnalysisResult:
        """Parse text and return an AnalysisResult (iterable, indexable).

        Args:
            text: Input text to analyze

        Returns:
            AnalysisResult containing Morpheme objects

        Raises:
            RuntimeError: If parsing fails

        Examples:
            >>> result = m.parse("東京は日本の首都です")
            >>> for morph in result:
            ...     print(morph.surface, morph.pos)
        """
        ...

    def parse_to_json(self, text: str) -> str:
        """Parse text and return a JSON string directly.

        Equivalent to ``m.parse(text).to_json()``.

        Args:
            text: Input text to analyze

        Returns:
            JSON string with ``text`` and ``morphemes`` fields

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    def wakati(self, text: str) -> str:
        """Parse text and return wakati (space-separated) output.

        Args:
            text: Input text to analyze

        Returns:
            Space-separated surface forms

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    def wakati_list(self, text: str) -> List[str]:
        """Parse text and return surface forms as a Python list.

        Args:
            text: Input text to analyze

        Returns:
            List of surface-form strings

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    # Legacy / convenience parse variants
    def parse_to_list(self, text: str) -> List[Tuple[str, str]]:
        """Parse text and return list of (surface, feature) tuples."""
        ...

    def parse_to_dict(self, text: str) -> List[Dict[str, Any]]:
        """Parse text and return list of morpheme dictionaries."""
        ...

    def parse_to_morphemes(self, text: str) -> List[Morpheme]:
        """Parse text and return list of Morpheme objects."""
        ...

    # N-best analysis
    def parse_nbest(
        self, text: str, n: int = 5
    ) -> List[Tuple[AnalysisResult, int]]:
        """Parse text and return N-best analysis results.

        Args:
            text: Input text to analyze
            n: Number of best paths to return (default: 5)

        Returns:
            List of (AnalysisResult, cost) tuples sorted by cost.
            Lower cost indicates more likely analysis.

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    def parse_nbest_reranked(
        self, text: str, n: int = 5, reranker: str = "null"
    ) -> AnalysisResult:
        """Parse text and return the reranker-selected best path.

        Args:
            text: Input text to analyze.
            n: Number of N-best candidates to generate (default: 5).
            reranker: Reranker strategy — ``"null"`` (default, always index 0)
                      or ``"cost"`` (minimum-cost candidate).

        Returns:
            AnalysisResult for the selected path.

        Raises:
            RuntimeError: If parsing fails or no candidates are produced.
        """
        ...

    def tokenize_text(self, text: str, vocab: SurfaceVocab) -> List[int]:
        """Parse text and return a list of word IDs using the provided vocab.

        New surface forms are automatically assigned the next available ID.

        Args:
            text: Input text to tokenize.
            vocab: A ``SurfaceVocab`` instance that maps surfaces to IDs.

        Returns:
            List of integer word IDs, one per morpheme in the analysis result.

        Raises:
            RuntimeError: If parsing fails or the vocab lock is poisoned.
        """
        ...

    # BPE / probabilistic output
    def parse_bpe(self, text: str) -> str:
        """Parse text and return in SentencePiece-compatible BPE format.

        Returns tokens with ▁ (U+2581) word-initial markers, space-separated.
        Compatible with SentencePiece and mT5 tokenizer training pipelines.

        Args:
            text: Input Japanese text

        Returns:
            BPE-compatible tokenization (e.g., "▁東京 は ▁日本 の ▁首都")

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    def parse_with_probs(
        self, text: str
    ) -> Tuple[AnalysisResult, List[Dict[str, Any]]]:
        """Parse text and return analysis with marginal probabilities.

        Uses the forward-backward algorithm to compute P(morpheme | input)
        for every morpheme in the optimal path.

        Args:
            text: Input Japanese text

        Returns:
            Tuple of (AnalysisResult, list of probability dicts).
            Each dict has: surface (str), log_prob (float), prob (float).

        Raises:
            RuntimeError: If parsing fails

        Example:
            >>> result, probs = mecrab.parse_with_probs("東京")
            >>> for p in probs:
            ...     print(f"{p['surface']}: {p['prob']:.3f}")
        """
        ...

    def parse_bpe_batch(self, texts: List[str]) -> List[str]:
        """Parse multiple texts in batch and return BPE-compatible format.

        Uses parallel processing (Rayon) internally for performance and
        releases the GIL so other Python threads can run concurrently.

        Args:
            texts: List of input Japanese texts

        Returns:
            List of BPE-formatted strings with ▁ word-initial markers,
            one per input text (e.g., ["▁東京 は ▁日本", "▁大阪 に ▁行く"])

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def parse_with_probs_batch(
        self, texts: List[str]
    ) -> List[Tuple[AnalysisResult, List[Dict[str, Any]]]]:
        """Parse multiple texts and return analysis with marginal probabilities.

        For each text, computes P(morpheme | input) for every lattice position
        via the forward-backward algorithm.  The GIL is released during the
        Rust computation phase.

        Args:
            texts: List of input texts

        Returns:
            List of (AnalysisResult, prob_list) tuples, one per input text.
            prob_list entries: {"surface": str, "log_prob": float, "prob": float}

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def parse_nbest_with_probs(
        self, text: str, n: int = 5
    ) -> List[Tuple[AnalysisResult, int, List[Dict[str, Any]]]]:
        """Parse text and return N-best segmentations with marginal probabilities.

        The N-best paths are computed via the Viterbi algorithm; the marginal
        probabilities are computed once from the full lattice via
        forward-backward and are therefore shared across all N-best paths.
        This is the correct interpretation: probs reflect P(morpheme | ALL paths),
        not P(morpheme | this specific N-best candidate).

        Args:
            text: Input Japanese text
            n: Number of N-best candidates to return (default: 5)

        Returns:
            List of (AnalysisResult, cost, prob_list) tuples in ascending cost order.
            cost: total Viterbi path cost (lower = more likely)
            prob_list: marginal probabilities shared across all N-best paths,
                       dicts with {"surface": str, "log_prob": float, "prob": float}

        Raises:
            RuntimeError: If parsing fails or the lattice produces no candidates
        """
        ...

    # JSON output
    def parse_json(self, text: str) -> str:
        """Parse text and return JSON array of morpheme objects."""
        ...

    def parse_jsonld(self, text: str) -> str:
        """Parse text and return JSON-LD output with semantic annotations."""
        ...

    # Batch processing
    def parse_batch(self, texts: List[str]) -> List[str]:
        """Parse multiple texts in batch, returning MeCab-format strings.

        When compiled with ``parallel`` feature, uses Rayon.

        Args:
            texts: List of texts to analyze

        Returns:
            List of MeCab-format strings

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def parse_batch_py(self, texts: List[str]) -> List[AnalysisResult]:
        """Parse multiple texts, returning AnalysisResult objects.

        Releases the GIL during batch processing.

        Args:
            texts: List of texts to analyze

        Returns:
            List of AnalysisResult objects

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def wakati_batch(self, texts: List[str]) -> List[str]:
        """Parse multiple texts and return wakati outputs."""
        ...

    def parse_batch_to_morphemes(self, texts: List[str]) -> List[List[Morpheme]]:
        """Parse multiple texts and return Morpheme objects in batch."""
        ...

    def parse_batch_to_dict(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """Parse multiple texts and return dictionaries in batch."""
        ...

    # Dictionary management
    def add_word(
        self, surface: str, reading: str, pronunciation: str, wcost: int
    ) -> None:
        """Add a word to the overlay dictionary.

        Args:
            surface: The surface form (the actual text)
            reading: The katakana reading
            pronunciation: The pronunciation (often same as reading)
            wcost: Word cost (lower = more preferred, typical: 5000-8000)
        """
        ...

    def remove_word(self, surface: str) -> bool:
        """Remove a word from the overlay dictionary.

        Returns:
            True if the word was found and removed, False otherwise
        """
        ...

    def overlay_size(self) -> int:
        """Get the number of words in the overlay dictionary."""
        ...

    def dict_info(self) -> Dict[str, Any]:
        """Get dictionary information (overlay_size, has_vectors, with_ipa)."""
        ...

    def is_loaded(self) -> bool:
        """Return True if the analyser is fully initialised and ready."""
        ...

    # IPA pronunciation
    def to_ipa(self, text: str) -> List[str]:
        """Convert text to IPA pronunciation list (requires with_ipa=True)."""
        ...

    def to_ipa_text(self, text: str, separator: str = " ") -> str:
        """Convert text to IPA pronunciation as a single string (requires with_ipa=True)."""
        ...

    # Word embeddings
    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words (requires vector_path)."""
        ...

    def most_similar(
        self, word: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """Find words most similar to the given word (requires vector_path)."""
        ...

    def analogy(
        self,
        positive1: str,
        negative: str,
        positive2: str,
        topn: int = 5,
    ) -> List[Tuple[str, float]]:
        """Perform word analogy: positive1 - negative + positive2 = ? (requires vector_path)."""
        ...

    def sentence_embedding(self, text: str) -> List[float]:
        """Get sentence embedding via mean pooling (requires vector_path)."""
        ...

    # Context manager
    def __enter__(self) -> "MeCrab": ...
    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool: ...


# Module-level functions
def version() -> str:
    """Get MeCrab version string (e.g., ``"0.2.0"``)."""
    ...


def default_dicdir() -> Optional[str]:
    """Get default IPADIC dictionary path, or None if not found.

    Examples:
        >>> import mecrab
        >>> dicdir = mecrab.default_dicdir()
        >>> if dicdir:
        ...     print(f"Dictionary at: {dicdir}")
    """
    ...


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity in range [-1.0, 1.0]

    Raises:
        ValueError: If vectors have different dimensions or are zero

    Examples:
        >>> import mecrab
        >>> mecrab.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        1.0
    """
    ...
