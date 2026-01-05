"""Type stubs for mecrab native module."""

from typing import Any, Dict, List, Optional, Tuple

class MeCrab:
    """High-performance morphological analyzer compatible with MeCab.

    Features:
    - Fast morphological analysis
    - IPA (International Phonetic Alphabet) pronunciation
    - Word embeddings with cosine similarity
    - Live dictionary updates
    - Batch processing with parallel support
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
            >>> # Basic usage
            >>> m = MeCrab()

            >>> # With IPA pronunciation
            >>> m = MeCrab(with_ipa=True)

            >>> # With word embeddings
            >>> m = MeCrab(vector_path="vectors.bin")

            >>> # Combined IPA + vectors
            >>> m = MeCrab(with_ipa=True, vector_path="vectors.bin")
        """
        ...

    def parse(self, text: str) -> str:
        """Parse text and return analysis result.

        Args:
            text: Input text to analyze

        Returns:
            Analysis result as formatted string (MeCab format)

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

    def parse_to_list(self, text: str) -> List[Tuple[str, str]]:
        """Parse text and return list of morphemes.

        Args:
            text: Input text to analyze

        Returns:
            List of (surface, feature) tuples

        Raises:
            RuntimeError: If parsing fails
        """
        ...

    def parse_to_dict(self, text: str) -> List[Dict[str, Any]]:
        """Parse text and return list of dictionaries (Pythonic API).

        Returns structured morpheme information including optional fields
        (IPA pronunciation, word embeddings) when enabled.

        Args:
            text: Input text to analyze

        Returns:
            List of dictionaries with morpheme information. Each dict contains:
            - surface (str): Surface form
            - feature (str): Full feature string
            - pos (str): Part-of-speech
            - pos1 (str, optional): POS subcategory 1
            - pos2 (str, optional): POS subcategory 2
            - pos3 (str, optional): POS subcategory 3
            - inflection (str, optional): Inflection type
            - conjugation (str, optional): Conjugation form
            - base (str, optional): Base form
            - reading (str, optional): Reading (katakana)
            - pronunciation (str, optional): Pronunciation (katakana)
            - ipa (str, optional): IPA pronunciation (if with_ipa=True)
            - embedding (List[float], optional): Word embedding vector (if vector_path provided)

        Raises:
            RuntimeError: If parsing fails

        Examples:
            >>> m = MeCrab()
            >>> result = m.parse_to_dict("東京に行く")
            >>> for morph in result:
            ...     print(morph['surface'], morph['pos'])
            東京 名詞
            に 助詞
            行く 動詞
        """
        ...

    def parse_batch(self, texts: List[str]) -> List[str]:
        """Parse multiple texts in batch.

        When compiled with 'parallel' feature, this uses Rayon for
        parallel processing across all available CPU cores.

        Args:
            texts: List of texts to analyze

        Returns:
            List of analysis results as formatted strings

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def wakati_batch(self, texts: List[str]) -> List[str]:
        """Parse multiple texts and return wakati outputs in batch.

        Args:
            texts: List of texts to analyze

        Returns:
            List of space-separated surface forms

        Raises:
            RuntimeError: If any parsing fails
        """
        ...

    def add_word(
        self, surface: str, reading: str, pronunciation: str, wcost: int
    ) -> None:
        """Add a word to the overlay dictionary.

        This allows adding custom words (new product names, slang, etc.)
        that will be recognized during parsing.

        Args:
            surface: The surface form (the actual text)
            reading: The katakana reading
            pronunciation: The pronunciation
            wcost: Word cost (lower = more preferred, typical: 5000-8000)
        """
        ...

    def remove_word(self, surface: str) -> bool:
        """Remove a word from the overlay dictionary.

        Args:
            surface: The surface form to remove

        Returns:
            True if the word was found and removed
        """
        ...

    def overlay_size(self) -> int:
        """Get the number of words in the overlay dictionary.

        Returns:
            Number of overlay words
        """
        ...

    def to_ipa(self, text: str) -> List[str]:
        """Convert text to IPA pronunciation (one-shot conversion).

        Requires: MeCrab initialized with with_ipa=True

        This is a convenience method that parses the text and returns
        just the IPA pronunciations as a list of strings.

        Args:
            text: Input text to convert

        Returns:
            List of IPA pronunciation strings (one per morpheme)

        Raises:
            RuntimeError: If IPA is not enabled or parsing fails

        Examples:
            >>> m = MeCrab(with_ipa=True)
            >>> ipas = m.to_ipa("東京に行く")
            >>> print(ipas)
            ['toːkʲoː', 'ɲi', 'ikɯ']

            >>> # Join with spaces
            >>> print(" ".join(ipas))
            'toːkʲoː ɲi ikɯ'
        """
        ...

    def to_ipa_text(self, text: str, separator: str = " ") -> str:
        """Convert text to IPA pronunciation as a single string.

        Requires: MeCrab initialized with with_ipa=True

        Args:
            text: Input text to convert
            separator: Separator between morphemes (default: " ")

        Returns:
            IPA pronunciation string

        Raises:
            RuntimeError: If IPA is not enabled or parsing fails

        Examples:
            >>> m = MeCrab(with_ipa=True)

            >>> # Default separator (space)
            >>> ipa = m.to_ipa_text("東京に行く")
            >>> print(ipa)
            'toːkʲoː ɲi ikɯ'

            >>> # Custom separator
            >>> print(m.to_ipa_text("東京に行く", separator="-"))
            'toːkʲoː-ɲi-ikɯ'
        """
        ...

    def similarity(self, word1: str, word2: str) -> float:
        """Compute cosine similarity between two words.

        Requires: MeCrab initialized with vector_path parameter

        Parses both words and computes the cosine similarity between their
        embedding vectors. If a word tokenizes into multiple morphemes,
        uses the first morpheme's embedding.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Cosine similarity in range [-1.0, 1.0]

        Raises:
            RuntimeError: If:
            - Vectors not enabled
            - Word not found in vocabulary
            - Word has no embedding (out-of-vocabulary)

        Examples:
            >>> m = MeCrab(vector_path="vectors.bin")

            >>> # Compare similar words
            >>> sim = m.similarity("東京", "京都")
            >>> print(f"Similarity: {sim:.3f}")
            Similarity: 0.856

            >>> # Compare dissimilar words
            >>> sim = m.similarity("東京", "食べる")
            >>> print(f"Similarity: {sim:.3f}")
            Similarity: 0.123
        """
        ...

class Morpheme:
    """A single morpheme from analysis."""

    surface: str
    """Surface form"""

    feature: str
    """Feature string"""

    pos_id: int
    """Part-of-speech ID"""

    wcost: int
    """Word cost"""

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

def version() -> str:
    """Get MeCrab version.

    Returns:
        Version string (e.g., "0.1.0")
    """
    ...
