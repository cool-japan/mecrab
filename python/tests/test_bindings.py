"""Tests for MeCrab Python bindings.

These tests require a real IPADIC dictionary to run.
They are skipped gracefully when the dictionary is not available.

Run with::

    MECRAB_DICT_PATH=/usr/lib/mecab/dic/ipadic pytest python/tests/

Or let mecrab auto-detect the dictionary::

    pytest python/tests/
"""
import json
import os
import tempfile

import pytest

# Allow override via environment variable; fall back to common locations.
_CANDIDATE_PATHS = [
    os.environ.get("MECRAB_DICT_PATH", ""),
    "/usr/lib/mecab/dic/ipadic",
    "/usr/lib/mecab/dic/ipadic-utf8",
    "/var/lib/mecab/dic/ipadic-utf8",
    "/usr/local/lib/mecab/dic/ipadic-utf8",
    "/opt/homebrew/lib/mecab/dic/ipadic-utf8",
    "/opt/homebrew/lib/mecab/dic/ipadic",
]

DICT_PATH: str = next(
    (p for p in _CANDIDATE_PATHS if p and os.path.isfile(os.path.join(p, "sys.dic"))),
    "",
)
HAS_DICT: bool = bool(DICT_PATH)

pytestmark = pytest.mark.skipif(
    not HAS_DICT, reason="IPADIC dictionary not found (set MECRAB_DICT_PATH)"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mecrab():
    from mecrab import MeCrab
    return MeCrab(DICT_PATH)


# ---------------------------------------------------------------------------
# TestMorpheme
# ---------------------------------------------------------------------------

class TestMorpheme:
    def test_repr_contains_surface(self, mecrab):
        result = mecrab.parse("東京")
        m = result[0]
        assert "東京" in repr(m)

    def test_repr_format(self, mecrab):
        result = mecrab.parse("東京")
        m = result[0]
        # Expected shape: Morpheme(surface='東京', pos='...', reading=...)
        assert repr(m).startswith("Morpheme(")
        assert "surface=" in repr(m)
        assert "pos=" in repr(m)

    def test_str_is_mecab_format(self, mecrab):
        result = mecrab.parse("東京")
        m = result[0]
        s = str(m)
        # surface\tfeature
        assert "\t" in s
        assert s.startswith("東京")

    def test_to_dict_has_required_keys(self, mecrab):
        result = mecrab.parse("東京")
        d = result[0].to_dict()
        assert isinstance(d, dict)
        assert "surface" in d
        assert "pos" in d
        assert "feature" in d
        assert d["surface"] == "東京"

    def test_to_json_is_valid_json(self, mecrab):
        result = mecrab.parse("東京")
        j = json.loads(result[0].to_json())
        assert isinstance(j, dict)
        assert j["surface"] == "東京"
        assert "pos" in j

    def test_to_json_escapes_special_chars(self, mecrab):
        # Use a text where we can validate JSON escaping safety
        result = mecrab.parse("テスト")
        raw = result[0].to_json()
        # Must parse without error
        parsed = json.loads(raw)
        assert "surface" in parsed

    def test_pos_helpers(self, mecrab):
        result = mecrab.parse("東京に行く")
        noun_found = any(m.is_noun() for m in result)
        verb_found = any(m.is_verb() for m in result)
        particle_found = any(m.is_particle() for m in result)
        assert noun_found, "Expected at least one noun"
        assert verb_found, "Expected at least one verb"
        assert particle_found, "Expected at least one particle"

    def test_wcost_is_integer(self, mecrab):
        result = mecrab.parse("東京")
        assert isinstance(result[0].wcost, int)

    def test_pos_id_is_integer(self, mecrab):
        result = mecrab.parse("東京")
        assert isinstance(result[0].pos_id, int)


# ---------------------------------------------------------------------------
# TestAnalysisResult
# ---------------------------------------------------------------------------

class TestAnalysisResult:
    def test_repr(self, mecrab):
        result = mecrab.parse("東京は日本の首都です")
        r = repr(result)
        assert "AnalysisResult" in r
        assert "東京は日本の首都です" in r

    def test_str_ends_with_eos(self, mecrab):
        result = mecrab.parse("東京")
        assert str(result).strip().endswith("EOS")

    def test_len_positive(self, mecrab):
        result = mecrab.parse("東京は日本の首都です")
        assert len(result) > 0

    def test_iter_yields_morphemes(self, mecrab):
        result = mecrab.parse("東京は日本の首都です")
        surfaces = [m.surface for m in result]
        assert "東京" in surfaces

    def test_iter_all_are_morpheme(self, mecrab):
        from mecrab import Morpheme
        result = mecrab.parse("東京")
        for m in result:
            assert isinstance(m, Morpheme)

    def test_getitem_first(self, mecrab):
        result = mecrab.parse("東京")
        assert result[0].surface == "東京"

    def test_getitem_negative(self, mecrab):
        result = mecrab.parse("東京")
        # result[-1] must be the last morpheme
        last_by_positive = result[len(result) - 1]
        last_by_negative = result[-1]
        assert last_by_positive.surface == last_by_negative.surface

    def test_getitem_out_of_range(self, mecrab):
        result = mecrab.parse("東京")
        with pytest.raises(IndexError):
            _ = result[999]

    def test_getitem_negative_out_of_range(self, mecrab):
        result = mecrab.parse("東京")
        with pytest.raises(IndexError):
            _ = result[-999]

    def test_to_list_is_list_of_dicts(self, mecrab):
        result = mecrab.parse("東京")
        lst = result.to_list()
        assert isinstance(lst, list)
        assert len(lst) > 0
        assert all(isinstance(d, dict) for d in lst)
        assert all("surface" in d for d in lst)

    def test_to_json_is_valid(self, mecrab):
        result = mecrab.parse("東京")
        parsed = json.loads(result.to_json())
        assert "text" in parsed
        assert "morphemes" in parsed
        assert parsed["text"] == "東京"
        assert isinstance(parsed["morphemes"], list)

    def test_surfaces(self, mecrab):
        result = mecrab.parse("東京は日本の首都です")
        surfaces = result.surfaces()
        assert isinstance(surfaces, list)
        assert "東京" in surfaces

    def test_readings(self, mecrab):
        result = mecrab.parse("東京")
        readings = result.readings()
        assert isinstance(readings, list)
        assert len(readings) == len(result)

    def test_pos_tags(self, mecrab):
        result = mecrab.parse("東京は日本の首都です")
        tags = result.pos_tags()
        assert isinstance(tags, list)
        assert len(tags) == len(result)
        assert "名詞" in tags

    def test_text_attribute(self, mecrab):
        text = "東京は日本の首都です"
        result = mecrab.parse(text)
        assert result.text == text

    def test_morphemes_attribute(self, mecrab):
        result = mecrab.parse("東京")
        assert isinstance(result.morphemes, list)
        assert len(result.morphemes) > 0


# ---------------------------------------------------------------------------
# TestMeCrab
# ---------------------------------------------------------------------------

class TestMeCrab:
    def test_is_loaded(self, mecrab):
        assert mecrab.is_loaded() is True

    def test_parse_returns_analysis_result(self, mecrab):
        from mecrab import AnalysisResult
        result = mecrab.parse("東京")
        assert isinstance(result, AnalysisResult)

    def test_parse_to_json(self, mecrab):
        j = mecrab.parse_to_json("東京")
        parsed = json.loads(j)
        assert "text" in parsed
        assert "morphemes" in parsed

    def test_wakati_returns_str(self, mecrab):
        w = mecrab.wakati("東京は日本の首都です")
        assert isinstance(w, str)
        assert "東京" in w

    def test_wakati_list_returns_list(self, mecrab):
        wl = mecrab.wakati_list("東京は日本の首都です")
        assert isinstance(wl, list)
        assert "東京" in wl

    def test_wakati_list_matches_wakati(self, mecrab):
        text = "東京は日本の首都です"
        wakati_str = mecrab.wakati(text)
        wakati_lst = mecrab.wakati_list(text)
        # wakati_list should be the same tokens as wakati split by space
        assert " ".join(wakati_lst) == wakati_str

    def test_parse_batch_py_returns_list(self, mecrab):
        texts = ["東京", "大阪", "京都"]
        results = mecrab.parse_batch_py(texts)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_parse_batch_py_results_match_single(self, mecrab):
        from mecrab import AnalysisResult
        texts = ["東京", "大阪"]
        batch = mecrab.parse_batch_py(texts)
        for i, text in enumerate(texts):
            single = mecrab.parse(text)
            assert isinstance(batch[i], AnalysisResult)
            assert batch[i].surfaces() == single.surfaces()

    def test_overlay_size_initial(self, mecrab):
        # Not testing exact value, just that it's an integer >= 0
        assert isinstance(mecrab.overlay_size(), int)
        assert mecrab.overlay_size() >= 0

    def test_context_manager(self):
        from mecrab import MeCrab
        with MeCrab(DICT_PATH) as m:
            result = m.parse("東京")
            assert len(result) > 0

    def test_has_ipa_false_by_default(self, mecrab):
        assert mecrab.has_ipa is False

    def test_has_vectors_false_by_default(self, mecrab):
        assert mecrab.has_vectors is False

    def test_dict_info(self, mecrab):
        info = mecrab.dict_info()
        assert isinstance(info, dict)
        assert "overlay_size" in info
        assert "has_vectors" in info
        assert "with_ipa" in info


# ---------------------------------------------------------------------------
# TestCustomWords
# ---------------------------------------------------------------------------

class TestCustomWords:
    def test_add_word_increases_overlay_size(self, mecrab):
        before = mecrab.overlay_size()
        mecrab.add_word("MeCrabテスト語", "メクラブテストゴ", "メクラブテストゴ", 5000)
        after = mecrab.overlay_size()
        assert after == before + 1
        # Clean up
        mecrab.remove_word("MeCrabテスト語")

    def test_remove_word_returns_true(self, mecrab):
        mecrab.add_word("MeCrabテスト語2", "メクラブテストゴツー", "メクラブテストゴツー", 5000)
        result = mecrab.remove_word("MeCrabテスト語2")
        assert result is True

    def test_remove_nonexistent_word_returns_false(self, mecrab):
        result = mecrab.remove_word("絶対に存在しない単語XYZ123")
        assert result is False


# ---------------------------------------------------------------------------
# TestVersion
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_is_string(self):
        from mecrab import version
        assert isinstance(version(), str)
        assert len(version()) > 0

    def test_module_version(self):
        import mecrab
        assert isinstance(mecrab.__version__, str)
        assert mecrab.__version__ == "0.2.0"

    def test_default_dicdir(self):
        from mecrab import default_dicdir
        result = default_dicdir()
        # Either None or a non-empty string
        assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# TestCosimeSimilarity (module-level function, no dict needed)
# ---------------------------------------------------------------------------

# These tests do not need a dictionary; remove the module-level skip
class TestCosineSimilarity:
    @pytest.mark.no_dict
    def test_identical_vectors(self):
        from mecrab import cosine_similarity
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)

    @pytest.mark.no_dict
    def test_orthogonal_vectors(self):
        from mecrab import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    @pytest.mark.no_dict
    def test_opposite_vectors(self):
        from mecrab import cosine_similarity
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    @pytest.mark.no_dict
    def test_dimension_mismatch_raises(self):
        from mecrab import cosine_similarity
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
