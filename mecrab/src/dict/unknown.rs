//! Unknown word dictionary handling (unk.dic)
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module handles unknown (out-of-vocabulary) words by loading
//! the unk.dic file which has the same format as sys.dic but contains
//! entries for each character category.
//!
//! Reference: ../ref/mecab-0.996/src/dictionary.cpp

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::Arc;

use super::DictionaryEntry;
use super::char_def::CharCategory;
use super::sys_dic::SysDic;

/// Map from `CharCategory` to the pre-built template entries for that category.
///
/// Each template `DictionaryEntry` has `length = 0`; callers clone the template
/// and overwrite `length` with the actual unknown-word byte length.
type CategoryTemplates = HashMap<CharCategory, Vec<DictionaryEntry>>;

/// Unknown word dictionary
///
/// This uses the same format as the system dictionary (sys.dic)
/// but contains entries keyed by category name (e.g., "HIRAGANA", "KATAKANA").
///
/// ## Template cache
///
/// At load time all 11 `CharCategory` variants are queried once and their
/// prototype `DictionaryEntry` lists are stored in `templates`.  Subsequent
/// calls to `generate_entries(category, length)` clone the cached slice and
/// set the `length` field — avoiding a `common_prefix_search` trie traversal
/// on every invocation.
pub struct UnknownDictionary {
    /// Underlying system dictionary structure
    inner: SysDic,
    /// Per-category prototype entries (length field = 0; filled at call time).
    templates: CategoryTemplates,
}

impl std::fmt::Debug for UnknownDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnknownDictionary")
            .field("inner", &self.inner)
            .finish()
    }
}

/// All 11 `CharCategory` variants in declaration order.
///
/// Used to iterate over the full set during template pre-computation.
const ALL_CATEGORIES: &[CharCategory] = &[
    CharCategory::Default,
    CharCategory::Space,
    CharCategory::Kanji,
    CharCategory::Symbol,
    CharCategory::Numeric,
    CharCategory::Alpha,
    CharCategory::Hiragana,
    CharCategory::Katakana,
    CharCategory::Kanjinumeric,
    CharCategory::Greek,
    CharCategory::Cyrillic,
];

/// Map a `CharCategory` to the canonical ASCII name used as a key in `unk.dic`.
#[inline]
fn category_name(category: CharCategory) -> &'static str {
    match category {
        CharCategory::Default => "DEFAULT",
        CharCategory::Space => "SPACE",
        CharCategory::Kanji => "KANJI",
        CharCategory::Symbol => "SYMBOL",
        CharCategory::Numeric => "NUMERIC",
        CharCategory::Alpha => "ALPHA",
        CharCategory::Hiragana => "HIRAGANA",
        CharCategory::Katakana => "KATAKANA",
        CharCategory::Kanjinumeric => "KANJINUMERIC",
        CharCategory::Greek => "GREEK",
        CharCategory::Cyrillic => "CYRILLIC",
    }
}

/// Build the template cache by querying `inner` once per category.
///
/// Template entries have `length = 0`.  The real byte-length is injected by
/// `generate_entries` at call time without touching the trie.
fn build_templates(inner: &SysDic) -> CategoryTemplates {
    let mut map = HashMap::with_capacity(ALL_CATEGORIES.len());
    for &cat in ALL_CATEGORIES {
        let name = category_name(cat);
        // Exact-match filter: keep only entries whose trie-matched length
        // equals the full category-name byte length.
        let mut entries: Vec<DictionaryEntry> = inner
            .common_prefix_search(name)
            .into_iter()
            .filter(|e| e.length == name.len())
            .collect();
        // Set length to sentinel 0; callers overwrite before use.
        for e in &mut entries {
            e.length = 0;
        }
        map.insert(cat, entries);
    }
    map
}

impl UnknownDictionary {
    /// Load unknown word dictionary from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupted.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let inner = SysDic::from_mmap(mmap)?;

        // Verify this is actually an unknown dictionary (type = 2)
        if inner.dict_type() != super::MECAB_UNK_DIC {
            return Err(Error::InvalidDictionaryFormat(format!(
                "Expected unknown dictionary (type=2), got type={}",
                inner.dict_type()
            )));
        }

        let templates = build_templates(&inner);
        Ok(Self { inner, templates })
    }

    /// Load unknown word dictionary from an owned byte buffer (no filesystem required).
    ///
    /// # Errors
    ///
    /// Returns an error if the data is corrupted or the dictionary type does not match.
    pub fn from_bytes_owned(data: Arc<Vec<u8>>) -> Result<Self> {
        let inner = SysDic::from_bytes_owned(data)?;

        if inner.dict_type() != super::MECAB_UNK_DIC {
            return Err(Error::InvalidDictionaryFormat(format!(
                "Expected unknown dictionary (type=2), got type={}",
                inner.dict_type()
            )));
        }

        let templates = build_templates(&inner);
        Ok(Self { inner, templates })
    }

    /// Look up entries for a category name
    ///
    /// The category name should match the names from char.def
    /// (e.g., "DEFAULT", "SPACE", "KANJI", "HIRAGANA", "KATAKANA", etc.)
    pub fn lookup(&self, category_name_str: &str) -> Vec<DictionaryEntry> {
        self.inner.common_prefix_search(category_name_str)
    }

    /// Get all tokens for a category
    ///
    /// Returns entries that match the exact category name.
    pub fn get_entries_for_category(&self, category_name_str: &str) -> Vec<DictionaryEntry> {
        // For unknown dictionary, we need exact match
        self.inner
            .common_prefix_search(category_name_str)
            .into_iter()
            .filter(|e| e.length == category_name_str.len())
            .collect()
    }

    /// Get the charset
    pub fn charset(&self) -> &str {
        self.inner.charset()
    }

    /// Generate entries for unknown words based on category
    ///
    /// This is used by the lattice builder to create nodes for unknown words.
    /// The entries are looked up by category name and the length is set to the
    /// actual surface length of the unknown word.
    ///
    /// ## Caching
    ///
    /// The trie search and filter are performed once per category at load time
    /// and stored in `self.templates`.  This method clones the template slice
    /// and stamps `length` onto each copy — avoiding redundant trie traversals
    /// for every call within a run of same-category characters.
    ///
    /// # Arguments
    ///
    /// * `category` - The character category
    /// * `length` - The length of the unknown word surface in bytes
    pub fn generate_entries(
        &self,
        category: CharCategory,
        length: usize,
    ) -> Vec<DictionaryEntry> {
        // Cache hit: clone prototype entries and inject the real byte-length.
        if let Some(templates) = self.templates.get(&category) {
            return templates
                .iter()
                .map(|e| {
                    let mut entry = e.clone();
                    entry.length = length;
                    entry
                })
                .collect();
        }

        // Fallback (should never happen for the 11 known categories, but
        // provides backward safety if a new variant is added without updating
        // `ALL_CATEGORIES`).
        let name = category_name(category);
        self.inner
            .common_prefix_search(name)
            .into_iter()
            .filter(|e| e.length == name.len())
            .map(|mut e| {
                e.length = length;
                e
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unknown_dic_type() {
        assert_eq!(super::super::MECAB_UNK_DIC, 2);
    }

    /// Verify the category-name helper covers all 11 variants and returns
    /// distinct non-empty strings (compile-time correctness check).
    #[test]
    fn test_category_name_coverage() {
        let mut seen = std::collections::HashSet::new();
        for &cat in ALL_CATEGORIES {
            let name = category_name(cat);
            assert!(!name.is_empty(), "category name must be non-empty");
            assert!(
                seen.insert(name),
                "category name '{}' is duplicated",
                name
            );
        }
        assert_eq!(seen.len(), 11, "expected exactly 11 distinct category names");
    }

    /// Golden-value test: verify that `generate_entries` with and without the
    /// template cache produce identical results for every category/length pair.
    ///
    /// When a real `unk.dic` is not installed this test is a no-op (skipped via
    /// early return) so it never fails in CI environments without IPADIC.
    #[test]
    fn test_generate_entries_matches_direct_lookup() {
        // Try to load a real unk.dic from standard IPADIC locations.
        let unk_path = [
            "/var/lib/mecab/dic/ipadic-utf8/unk.dic",
            "/usr/lib/mecab/dic/ipadic-utf8/unk.dic",
            "/usr/local/lib/mecab/dic/ipadic-utf8/unk.dic",
            "/usr/share/mecab/dic/ipadic-utf8/unk.dic",
        ];

        let dict_opt = unk_path.iter().find_map(|p| {
            let path = std::path::Path::new(p);
            if path.exists() {
                let file = std::fs::File::open(path).ok()?;
                let mmap = Arc::new(unsafe { memmap2::Mmap::map(&file).ok()? });
                UnknownDictionary::from_mmap(mmap).ok()
            } else {
                None
            }
        });

        let Some(dict) = dict_opt else {
            eprintln!("No unk.dic found — skipping generate_entries golden-value test");
            return;
        };

        // For each category and a selection of lengths, verify that the cached
        // path returns byte-for-byte identical results to the direct-lookup path.
        for &cat in ALL_CATEGORIES {
            for &length in &[1usize, 3, 6, 9] {
                let via_cache = dict.generate_entries(cat, length);

                // Replicate the old (non-cached) logic inline for comparison.
                let name = category_name(cat);
                let direct: Vec<DictionaryEntry> = dict
                    .inner
                    .common_prefix_search(name)
                    .into_iter()
                    .filter(|e| e.length == name.len())
                    .map(|mut e| {
                        e.length = length;
                        e
                    })
                    .collect();

                assert_eq!(
                    via_cache.len(),
                    direct.len(),
                    "category {:?} length {} — entry count mismatch (cache={}, direct={})",
                    cat,
                    length,
                    via_cache.len(),
                    direct.len(),
                );

                for (i, (c, d)) in via_cache.iter().zip(direct.iter()).enumerate() {
                    assert_eq!(
                        c.length, d.length,
                        "entry[{}] length mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.left_id, d.left_id,
                        "entry[{}] left_id mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.right_id, d.right_id,
                        "entry[{}] right_id mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.wcost, d.wcost,
                        "entry[{}] wcost mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.feature, d.feature,
                        "entry[{}] feature mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.word_id, d.word_id,
                        "entry[{}] word_id mismatch for {:?}/{}",
                        i, cat, length
                    );
                    assert_eq!(
                        c.pos_id, d.pos_id,
                        "entry[{}] pos_id mismatch for {:?}/{}",
                        i, cat, length
                    );
                }
            }
        }
    }
}
