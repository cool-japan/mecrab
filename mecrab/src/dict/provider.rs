//! Dictionary provider trait abstraction for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module defines the `DictionaryProvider` trait that abstracts over
//! different dictionary format backends (IPADIC, UniDic, NEologd, etc.),
//! enabling format-agnostic morphological analysis.

/// Structured morpheme features extracted from a raw feature string.
///
/// This is a format-agnostic representation that maps the provider-specific
/// comma-separated fields to well-known semantic roles.
#[derive(Debug, Clone)]
pub struct MorphemeFeatures<'a> {
    /// Primary part of speech (e.g., "名詞", "動詞")
    pub pos: &'a str,
    /// POS sub-category 1 (e.g., "固有名詞", "自立")
    pub pos_detail1: Option<&'a str>,
    /// POS sub-category 2
    pub pos_detail2: Option<&'a str>,
    /// POS sub-category 3
    pub pos_detail3: Option<&'a str>,
    /// Conjugation type (活用型, e.g., "五段・カ行イ音便")
    pub conjugation_type: Option<&'a str>,
    /// Conjugation form (活用形, e.g., "連用形")
    pub conjugation_form: Option<&'a str>,
    /// Base (dictionary) form of the word
    pub base_form: Option<&'a str>,
    /// Reading in katakana
    pub reading: Option<&'a str>,
    /// Pronunciation in katakana (may differ from reading for particles)
    pub pronunciation: Option<&'a str>,
    /// All raw fields for pass-through access
    pub raw_fields: Vec<&'a str>,
}

impl<'a> MorphemeFeatures<'a> {
    /// Return `None` for an asterisk wildcard field used by MeCab dictionaries
    #[inline]
    fn opt(val: &'a str) -> Option<&'a str> {
        if val == "*" || val.is_empty() {
            None
        } else {
            Some(val)
        }
    }

    /// Build a `MorphemeFeatures` by splitting `feature_str` on commas.
    ///
    /// `raw_fields` will borrow from the split iterator — the caller must
    /// hold `feature_str` alive for the lifetime `'a`.
    ///
    /// Field indices are provided as named parameters so that each provider
    /// can call this single helper without duplicating the split logic.
    #[allow(clippy::too_many_arguments)]
    fn from_split(
        raw_fields: Vec<&'a str>,
        pos_idx: usize,
        pos_detail1_idx: Option<usize>,
        pos_detail2_idx: Option<usize>,
        pos_detail3_idx: Option<usize>,
        conjugation_type_idx: Option<usize>,
        conjugation_form_idx: Option<usize>,
        base_form_idx: Option<usize>,
        reading_idx: Option<usize>,
        pronunciation_idx: Option<usize>,
    ) -> Self {
        let get = |idx: usize| -> &'a str { raw_fields.get(idx).copied().unwrap_or("*") };
        let get_opt =
            |idx_opt: Option<usize>| -> Option<&'a str> { idx_opt.and_then(|i| Self::opt(get(i))) };

        let pos = get(pos_idx);

        Self {
            pos,
            pos_detail1: get_opt(pos_detail1_idx),
            pos_detail2: get_opt(pos_detail2_idx),
            pos_detail3: get_opt(pos_detail3_idx),
            conjugation_type: get_opt(conjugation_type_idx),
            conjugation_form: get_opt(conjugation_form_idx),
            base_form: get_opt(base_form_idx),
            reading: get_opt(reading_idx),
            pronunciation: get_opt(pronunciation_idx),
            raw_fields,
        }
    }
}

/// Trait abstracting over dictionary format backends (IPADIC, UniDic, NEologd, etc.)
///
/// Implement this trait to teach MeCrab how to interpret the feature string
/// produced by a particular dictionary format.  The trait is object-safe so
/// that it can be stored as `Arc<dyn DictionaryProvider>`.
pub trait DictionaryProvider: Send + Sync {
    /// Dictionary name/version string used for identification and debugging
    fn name(&self) -> &'static str;

    /// Number of feature fields per dictionary entry (e.g., 9 for IPADIC)
    fn feature_count(&self) -> usize;

    /// Parse a raw comma-separated feature string into a structured
    /// `MorphemeFeatures`.
    ///
    /// The lifetime `'a` ties the returned references to the input string so
    /// that zero-copy parsing is possible.
    fn parse_features<'a>(&self, feature_str: &'a str) -> MorphemeFeatures<'a>;

    /// Field index (0-based) for the primary part-of-speech token.
    /// Defaults to 0, which is correct for IPADIC, NEologd, and UniDic.
    fn pos_field(&self) -> usize {
        0
    }

    /// Field index for the katakana reading, or `None` if not present.
    fn reading_field(&self) -> Option<usize>;

    /// Field index for the base (dictionary) form, or `None` if not present.
    fn base_form_field(&self) -> Option<usize>;

    /// Field index for the pronunciation, or `None` if not present.
    fn pronunciation_field(&self) -> Option<usize>;
}

// ─────────────────────────────────────────────────────────────────────────────
// IPADIC provider
// ─────────────────────────────────────────────────────────────────────────────

/// IPADIC dictionary provider (9 fields).
///
/// Field layout (0-indexed):
/// ```text
/// 0  pos        (品詞)
/// 1  pos1       (品詞細分類1)
/// 2  pos2       (品詞細分類2)
/// 3  pos3       (品詞細分類3)
/// 4  cType      (活用型)
/// 5  cForm      (活用形)
/// 6  base_form  (原形)
/// 7  reading    (読み, katakana)
/// 8  pron       (発音, katakana)
/// ```
#[derive(Debug, Clone, Default)]
pub struct IpadicProvider;

impl DictionaryProvider for IpadicProvider {
    fn name(&self) -> &'static str {
        "IPADIC"
    }

    fn feature_count(&self) -> usize {
        9
    }

    fn reading_field(&self) -> Option<usize> {
        Some(7)
    }

    fn base_form_field(&self) -> Option<usize> {
        Some(6)
    }

    fn pronunciation_field(&self) -> Option<usize> {
        Some(8)
    }

    fn parse_features<'a>(&self, feature_str: &'a str) -> MorphemeFeatures<'a> {
        let raw_fields: Vec<&'a str> = feature_str.split(',').collect();
        MorphemeFeatures::from_split(
            raw_fields,
            0,       // pos
            Some(1), // pos_detail1
            Some(2), // pos_detail2
            Some(3), // pos_detail3
            Some(4), // conjugation_type
            Some(5), // conjugation_form
            Some(6), // base_form
            Some(7), // reading
            Some(8), // pronunciation
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// UniDic provider
// ─────────────────────────────────────────────────────────────────────────────

/// UniDic 2.x dictionary provider (28 fields).
///
/// Field layout (0-indexed):
/// ```text
///  0  pos1        (品詞)
///  1  pos2        (品詞細分類1)
///  2  pos3        (品詞細分類2)
///  3  pos4        (品詞細分類3)
///  4  cType       (活用型)
///  5  cForm       (活用形)
///  6  lForm       (語彙素読み)
///  7  lemma       (語彙素)
///  8  orth        (書字形出現形)
///  9  pron        (発音形出現形)
/// 10  orthBase    (書字形基本形)
/// 11  pronBase    (発音形基本形)
/// 12  goshu       (語種)
/// 13  iType       (語頭変化型)
/// 14  iForm       (語頭変化形)
/// 15  fType       (語末変化型)
/// 16  fForm       (語末変化形)
/// 17  iConType    (語頭変化結合型)
/// 18  fConType    (語末変化結合型)
/// 19  type        (アクセント型)
/// 20  kana        (仮名形出現形)
/// 21  kanaBase    (仮名形基本形)
/// 22  form        (語形出現形)
/// 23  formBase    (語形基本形)
/// 24  aType       (アクセント型)
/// 25  aConType    (アクセント結合型)
/// 26  aModType    (アクセント修飾型)
/// 27  lid         (語彙素ID)
/// ```
#[derive(Debug, Clone, Default)]
pub struct UnidicProvider;

impl DictionaryProvider for UnidicProvider {
    fn name(&self) -> &'static str {
        "UniDic"
    }

    fn feature_count(&self) -> usize {
        28
    }

    fn reading_field(&self) -> Option<usize> {
        Some(20) // kana
    }

    fn base_form_field(&self) -> Option<usize> {
        Some(10) // orthBase
    }

    fn pronunciation_field(&self) -> Option<usize> {
        Some(9) // pron
    }

    fn parse_features<'a>(&self, feature_str: &'a str) -> MorphemeFeatures<'a> {
        let raw_fields: Vec<&'a str> = feature_str.split(',').collect();
        // In UniDic the conjugation type is at index 4 and conjugation form at 5.
        // "base form" is the orthBase (field 10).
        // "reading" maps to kana (field 20) to be consistent with IPADIC semantics.
        MorphemeFeatures::from_split(
            raw_fields,
            0,        // pos (pos1)
            Some(1),  // pos_detail1 (pos2)
            Some(2),  // pos_detail2 (pos3)
            Some(3),  // pos_detail3 (pos4)
            Some(4),  // conjugation_type (cType)
            Some(5),  // conjugation_form (cForm)
            Some(10), // base_form (orthBase)
            Some(20), // reading (kana)
            Some(9),  // pronunciation (pron)
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// NEologd provider
// ─────────────────────────────────────────────────────────────────────────────

/// NEologd dictionary provider.
///
/// NEologd uses the same 9-field IPADIC layout but is distributed with a much
/// larger, more up-to-date lexicon.  Feature parsing is identical to IPADIC,
/// but the name is different so that callers can distinguish them in logs and
/// diagnostics.
#[derive(Debug, Clone, Default)]
pub struct NeologdProvider;

impl DictionaryProvider for NeologdProvider {
    fn name(&self) -> &'static str {
        "NEologd"
    }

    fn feature_count(&self) -> usize {
        IpadicProvider.feature_count()
    }

    fn reading_field(&self) -> Option<usize> {
        IpadicProvider.reading_field()
    }

    fn base_form_field(&self) -> Option<usize> {
        IpadicProvider.base_form_field()
    }

    fn pronunciation_field(&self) -> Option<usize> {
        IpadicProvider.pronunciation_field()
    }

    fn parse_features<'a>(&self, feature_str: &'a str) -> MorphemeFeatures<'a> {
        IpadicProvider.parse_features(feature_str)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-detect provider
// ─────────────────────────────────────────────────────────────────────────────

/// Auto-detecting dictionary provider that wraps another provider chosen at
/// construction time based on the observed feature count.
///
/// Use `AutoDetectProvider::detect(feature_count)` to create an instance.
pub struct AutoDetectProvider(Box<dyn DictionaryProvider>);

impl AutoDetectProvider {
    /// Choose the best-matching provider based on the number of feature fields
    /// found in the dictionary's feature table.
    ///
    /// | `feature_count` | Chosen provider |
    /// |-----------------|-----------------|
    /// | 28              | `UnidicProvider` |
    /// | 9               | `IpadicProvider` |
    /// | _other_         | `IpadicProvider` (safe fallback) |
    pub fn detect(feature_count: usize) -> Self {
        match feature_count {
            28 => Self(Box::new(UnidicProvider)),
            9 => Self(Box::new(IpadicProvider)),
            _ => Self(Box::new(IpadicProvider)), // safe fallback
        }
    }

    /// Wrap an arbitrary provider (useful for testing or custom formats).
    pub fn wrap(provider: impl DictionaryProvider + 'static) -> Self {
        Self(Box::new(provider))
    }
}

impl DictionaryProvider for AutoDetectProvider {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn feature_count(&self) -> usize {
        self.0.feature_count()
    }

    fn reading_field(&self) -> Option<usize> {
        self.0.reading_field()
    }

    fn base_form_field(&self) -> Option<usize> {
        self.0.base_form_field()
    }

    fn pronunciation_field(&self) -> Option<usize> {
        self.0.pronunciation_field()
    }

    fn parse_features<'a>(&self, feature_str: &'a str) -> MorphemeFeatures<'a> {
        self.0.parse_features(feature_str)
    }
}

// Manual Debug/Clone for AutoDetectProvider because Box<dyn …> is not Clone.
impl std::fmt::Debug for AutoDetectProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("AutoDetectProvider")
            .field(&self.0.name())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DictionaryFormat enum
// ─────────────────────────────────────────────────────────────────────────────

/// Hint for the dictionary format expected by the caller.
///
/// This is a lightweight enum that can be stored in configuration structs and
/// later converted into a concrete `Arc<dyn DictionaryProvider>` via
/// `DictionaryFormat::into_provider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DictionaryFormat {
    /// IPADIC format (9 fields) — most common MeCab dictionary
    #[default]
    Ipadic,
    /// UniDic 2.x format (28 fields)
    Unidic,
    /// NEologd extension of IPADIC (9 fields, larger lexicon)
    Neologd,
    /// Automatically detect from observed feature count
    Auto,
}

impl DictionaryFormat {
    /// Convert this format hint into a heap-allocated provider.
    ///
    /// When `Auto` is chosen, the provider resolves to IPADIC because the
    /// feature count is not known at this point.  Use
    /// `AutoDetectProvider::detect(n)` directly if you already know the count.
    #[must_use]
    pub fn into_provider(self) -> Box<dyn DictionaryProvider> {
        match self {
            Self::Ipadic => Box::new(IpadicProvider),
            Self::Unidic => Box::new(UnidicProvider),
            Self::Neologd => Box::new(NeologdProvider),
            Self::Auto => Box::new(AutoDetectProvider::detect(9)), // default to IPADIC
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── IPADIC ────────────────────────────────────────────────────────────────

    #[test]
    fn test_ipadic_provider_name_and_counts() {
        let p = IpadicProvider;
        assert_eq!(p.name(), "IPADIC");
        assert_eq!(p.feature_count(), 9);
        assert_eq!(p.pos_field(), 0);
        assert_eq!(p.reading_field(), Some(7));
        assert_eq!(p.base_form_field(), Some(6));
        assert_eq!(p.pronunciation_field(), Some(8));
    }

    #[test]
    fn test_ipadic_parse_noun() {
        let p = IpadicProvider;
        // Typical IPADIC noun: 東京
        let feat = "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.pos_detail1, Some("固有名詞"));
        assert_eq!(mf.pos_detail2, Some("地域"));
        assert_eq!(mf.pos_detail3, Some("一般"));
        assert_eq!(mf.conjugation_type, None); // *
        assert_eq!(mf.conjugation_form, None); // *
        assert_eq!(mf.base_form, Some("東京"));
        assert_eq!(mf.reading, Some("トウキョウ"));
        assert_eq!(mf.pronunciation, Some("トウキョウ"));
        assert_eq!(mf.raw_fields.len(), 9);
    }

    #[test]
    fn test_ipadic_parse_verb() {
        let p = IpadicProvider;
        // Typical IPADIC verb: 食べ
        let feat = "動詞,自立,*,*,一段,連用形,食べる,タベ,タベ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "動詞");
        assert_eq!(mf.pos_detail1, Some("自立"));
        assert_eq!(mf.pos_detail2, None); // *
        assert_eq!(mf.pos_detail3, None); // *
        assert_eq!(mf.conjugation_type, Some("一段"));
        assert_eq!(mf.conjugation_form, Some("連用形"));
        assert_eq!(mf.base_form, Some("食べる"));
        assert_eq!(mf.reading, Some("タベ"));
        assert_eq!(mf.pronunciation, Some("タベ"));
    }

    #[test]
    fn test_ipadic_parse_particle_wa() {
        let p = IpadicProvider;
        // Particle は — pronunciation differs from reading
        let feat = "助詞,係助詞,*,*,*,*,は,ハ,ワ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "助詞");
        assert_eq!(mf.reading, Some("ハ"));
        assert_eq!(mf.pronunciation, Some("ワ"));
    }

    #[test]
    fn test_ipadic_parse_short_feature_no_panic() {
        let p = IpadicProvider;
        // Only 3 fields — must not panic or unwrap
        let feat = "名詞,一般,*";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.pos_detail1, Some("一般"));
        assert_eq!(mf.pos_detail2, None); // *
        // Fields beyond the string should return None, not panic
        assert!(mf.base_form.is_none());
        assert!(mf.reading.is_none());
        assert!(mf.pronunciation.is_none());
    }

    // ── UniDic ────────────────────────────────────────────────────────────────

    #[test]
    fn test_unidic_provider_name_and_counts() {
        let p = UnidicProvider;
        assert_eq!(p.name(), "UniDic");
        assert_eq!(p.feature_count(), 28);
        assert_eq!(p.pos_field(), 0);
        assert_eq!(p.reading_field(), Some(20));
        assert_eq!(p.base_form_field(), Some(10));
        assert_eq!(p.pronunciation_field(), Some(9));
    }

    #[test]
    fn test_unidic_parse_28_fields() {
        let p = UnidicProvider;
        // Build a synthetic 28-field UniDic feature string
        // Fields: pos1,pos2,pos3,pos4,cType,cForm,lForm,lemma,orth,pron,
        //         orthBase,pronBase,goshu,iType,iForm,fType,fForm,iConType,
        //         fConType,type,kana,kanaBase,form,formBase,aType,aConType,
        //         aModType,lid
        let feat = "名詞,固有名詞,地名,一般,*,*,トウキョウ,東京,東京,トーキョー,\
                    東京,トーキョー,和,*,*,*,*,*,*,*,トウキョウ,トウキョウ,東京,東京,\
                    1,*,*,12345";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.pos_detail1, Some("固有名詞"));
        assert_eq!(mf.conjugation_type, None); // *
        assert_eq!(mf.base_form, Some("東京")); // orthBase (field 10)
        assert_eq!(mf.reading, Some("トウキョウ")); // kana (field 20)
        assert_eq!(mf.pronunciation, Some("トーキョー")); // pron (field 9)
        assert_eq!(mf.raw_fields.len(), 28);
    }

    #[test]
    fn test_unidic_parse_incomplete_fields() {
        let p = UnidicProvider;
        // Only 5 fields — must not panic
        let feat = "動詞,一般,*,*,五段-カ行";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "動詞");
        assert!(mf.base_form.is_none()); // field 10 absent
        assert!(mf.reading.is_none()); // field 20 absent
    }

    // ── NEologd ───────────────────────────────────────────────────────────────

    #[test]
    fn test_neologd_provider_name_and_counts() {
        let p = NeologdProvider;
        assert_eq!(p.name(), "NEologd");
        assert_eq!(p.feature_count(), 9);
        assert_eq!(p.reading_field(), Some(7));
        assert_eq!(p.base_form_field(), Some(6));
        assert_eq!(p.pronunciation_field(), Some(8));
    }

    #[test]
    fn test_neologd_parse_same_as_ipadic() {
        let ipadic = IpadicProvider;
        let neologd = NeologdProvider;
        let feat = "名詞,固有名詞,一般,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー";
        let mf_i = ipadic.parse_features(feat);
        let mf_n = neologd.parse_features(feat);
        assert_eq!(mf_i.pos, mf_n.pos);
        assert_eq!(mf_i.reading, mf_n.reading);
        assert_eq!(mf_i.base_form, mf_n.base_form);
    }

    // ── AutoDetect ────────────────────────────────────────────────────────────

    #[test]
    fn test_auto_detect_28_is_unidic() {
        let p = AutoDetectProvider::detect(28);
        assert_eq!(p.name(), "UniDic");
        assert_eq!(p.feature_count(), 28);
    }

    #[test]
    fn test_auto_detect_9_is_ipadic() {
        let p = AutoDetectProvider::detect(9);
        assert_eq!(p.name(), "IPADIC");
        assert_eq!(p.feature_count(), 9);
    }

    #[test]
    fn test_auto_detect_fallback() {
        let p = AutoDetectProvider::detect(12);
        assert_eq!(p.name(), "IPADIC"); // fallback
    }

    #[test]
    fn test_auto_detect_delegates_parse() {
        let p = AutoDetectProvider::detect(9);
        let feat = "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.reading, Some("トウキョウ"));
    }

    // ── DictionaryFormat ──────────────────────────────────────────────────────

    #[test]
    fn test_format_into_provider() {
        assert_eq!(DictionaryFormat::Ipadic.into_provider().name(), "IPADIC");
        assert_eq!(DictionaryFormat::Unidic.into_provider().name(), "UniDic");
        assert_eq!(DictionaryFormat::Neologd.into_provider().name(), "NEologd");
        assert_eq!(DictionaryFormat::Auto.into_provider().name(), "IPADIC");
    }

    #[test]
    fn test_format_default_is_ipadic() {
        let fmt = DictionaryFormat::default();
        assert_eq!(fmt, DictionaryFormat::Ipadic);
    }

    // ── Arc<dyn DictionaryProvider> (Send + Sync) ─────────────────────────────

    #[test]
    fn test_arc_dyn_provider_is_send_sync() {
        use std::sync::Arc;
        let p: Arc<dyn DictionaryProvider> = Arc::new(IpadicProvider);
        // Compile-time check: spawn a thread that holds the Arc
        let p2 = Arc::clone(&p);
        let handle = std::thread::spawn(move || {
            let feat = "名詞,一般,*,*,*,*,テスト,テスト,テスト";
            let mf = p2.parse_features(feat);
            mf.pos.len()
        });
        let _ = handle.join();
    }

    // ── Named auto-detect tests required by the loading pipeline ─────────────

    /// Auto-detect with 9 fields must resolve to IPADIC.
    #[test]
    fn test_auto_detect_9_fields() {
        let p = AutoDetectProvider::detect(9);
        assert_eq!(p.name(), "IPADIC");
        assert_eq!(p.feature_count(), 9);
        // Verify the provider can actually parse a 9-field IPADIC string.
        let feat = "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.reading, Some("トウキョウ"));
        assert_eq!(mf.raw_fields.len(), 9);
    }

    /// Auto-detect with 28 fields must resolve to UniDic.
    #[test]
    fn test_auto_detect_28_fields() {
        let p = AutoDetectProvider::detect(28);
        assert_eq!(p.name(), "UniDic");
        assert_eq!(p.feature_count(), 28);
        // Verify the provider can actually parse a 28-field UniDic string.
        let feat = "名詞,固有名詞,地名,一般,*,*,トウキョウ,東京,東京,トーキョー,\
                    東京,トーキョー,和,*,*,*,*,*,*,*,トウキョウ,トウキョウ,東京,東京,\
                    1,*,*,12345";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.base_form, Some("東京")); // orthBase at field 10
        assert_eq!(mf.reading, Some("トウキョウ")); // kana at field 20
        assert_eq!(mf.raw_fields.len(), 28);
    }

    /// Auto-detect with an unknown field count must fall back to IPADIC.
    #[test]
    fn test_auto_detect_unknown_fields() {
        // Field counts that are neither 9 nor 28 should safely fall back.
        for &unknown_count in &[0usize, 1, 5, 13, 17, 22, 100] {
            let p = AutoDetectProvider::detect(unknown_count);
            assert_eq!(
                p.name(),
                "IPADIC",
                "expected IPADIC fallback for field count {}",
                unknown_count
            );
        }
    }

    // ── MorphemeFeatures::opt helper ──────────────────────────────────────────

    #[test]
    fn test_opt_asterisk_returns_none() {
        assert!(MorphemeFeatures::opt("*").is_none());
        assert!(MorphemeFeatures::opt("").is_none());
        assert_eq!(MorphemeFeatures::opt("名詞"), Some("名詞"));
    }

    // ── Additional provider tests ─────────────────────────────────────────────

    #[test]
    fn test_parse_empty_feature_string() {
        let p = IpadicProvider;
        // An empty feature string must not panic; pos should be empty or "*"
        let mf = p.parse_features("");
        // splitting "" on ',' yields one element: ""
        assert!(mf.pos.is_empty() || mf.pos == "*");
        assert_eq!(mf.reading, None);
        assert_eq!(mf.base_form, None);
        assert_eq!(mf.pronunciation, None);
        // raw_fields should have exactly one element (the empty slice from split)
        assert_eq!(mf.raw_fields.len(), 1);
    }

    #[test]
    fn test_autodetect_wrap_constructor() {
        // AutoDetectProvider::wrap should accept any DictionaryProvider
        let p = AutoDetectProvider::wrap(NeologdProvider);
        assert_eq!(p.name(), "NEologd");
        assert_eq!(p.feature_count(), 9);
        assert_eq!(p.reading_field(), Some(7));
        let feat = "名詞,固有名詞,一般,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos, "名詞");
        assert_eq!(mf.base_form, Some("ChatGPT"));
        assert_eq!(mf.reading, Some("チャットジーピーティー"));
    }

    #[test]
    fn test_neologd_feature_count_equals_ipadic() {
        // NEologd delegates to IPADIC — feature counts must be identical
        assert_eq!(
            NeologdProvider.feature_count(),
            IpadicProvider.feature_count(),
            "NEologd and IPADIC must share the same feature count"
        );
    }

    #[test]
    fn test_pos_field_default_is_zero() {
        // All concrete providers must return 0 for the POS field (default impl)
        assert_eq!(IpadicProvider.pos_field(), 0);
        assert_eq!(UnidicProvider.pos_field(), 0);
        assert_eq!(NeologdProvider.pos_field(), 0);
        assert_eq!(AutoDetectProvider::detect(9).pos_field(), 0);
        assert_eq!(AutoDetectProvider::detect(28).pos_field(), 0);
    }

    #[test]
    fn test_ipadic_parse_asterisk_middle_fields_become_none() {
        // pos_detail2 and pos_detail3 are "*" while detail1 is present
        let p = IpadicProvider;
        let feat = "名詞,一般,*,*,*,*,東京,トウキョウ,トウキョウ";
        let mf = p.parse_features(feat);
        assert_eq!(mf.pos_detail1, Some("一般"));
        assert_eq!(mf.pos_detail2, None);
        assert_eq!(mf.pos_detail3, None);
        assert_eq!(mf.conjugation_type, None);
        assert_eq!(mf.conjugation_form, None);
        assert_eq!(mf.base_form, Some("東京"));
        assert_eq!(mf.reading, Some("トウキョウ"));
    }

    #[test]
    fn test_autodetect_detect_9_parse_delegates_to_ipadic() {
        // Ensure detect(9) parses identically to a bare IpadicProvider
        let auto = AutoDetectProvider::detect(9);
        let bare = IpadicProvider;
        let feat = "助詞,係助詞,*,*,*,*,は,ハ,ワ";
        let mf_auto = auto.parse_features(feat);
        let mf_bare = bare.parse_features(feat);
        assert_eq!(mf_auto.pos, mf_bare.pos);
        assert_eq!(mf_auto.reading, mf_bare.reading);
        assert_eq!(mf_auto.pronunciation, mf_bare.pronunciation);
        assert_eq!(mf_auto.raw_fields.len(), mf_bare.raw_fields.len());
    }

    #[test]
    fn test_autodetect_detect_28_parse_delegates_to_unidic() {
        // Ensure detect(28) parses identically to a bare UnidicProvider
        let auto = AutoDetectProvider::detect(28);
        let bare = UnidicProvider;
        let feat = "名詞,固有名詞,地名,一般,*,*,トウキョウ,東京,東京,トーキョー,\
                    東京,トーキョー,和,*,*,*,*,*,*,*,トウキョウ,トウキョウ,東京,東京,\
                    1,*,*,12345";
        let mf_auto = auto.parse_features(feat);
        let mf_bare = bare.parse_features(feat);
        assert_eq!(mf_auto.pos, mf_bare.pos);
        assert_eq!(mf_auto.reading, mf_bare.reading);
        assert_eq!(mf_auto.base_form, mf_bare.base_form);
        assert_eq!(mf_auto.pronunciation, mf_bare.pronunciation);
        assert_eq!(mf_auto.raw_fields.len(), mf_bare.raw_fields.len());
    }
}
