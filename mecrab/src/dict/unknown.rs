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
use std::sync::Arc;

use super::DictionaryEntry;
use super::sys_dic::SysDic;

/// Unknown word dictionary
///
/// This uses the same format as the system dictionary (sys.dic)
/// but contains entries keyed by category name (e.g., "HIRAGANA", "KATAKANA").
pub struct UnknownDictionary {
    /// Underlying system dictionary structure
    inner: SysDic,
}

impl std::fmt::Debug for UnknownDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnknownDictionary")
            .field("inner", &self.inner)
            .finish()
    }
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

        Ok(Self { inner })
    }

    /// Look up entries for a category name
    ///
    /// The category name should match the names from char.def
    /// (e.g., "DEFAULT", "SPACE", "KANJI", "HIRAGANA", "KATAKANA", etc.)
    pub fn lookup(&self, category_name: &str) -> Vec<DictionaryEntry> {
        self.inner.common_prefix_search(category_name)
    }

    /// Get all tokens for a category
    ///
    /// Returns entries that match the exact category name.
    pub fn get_entries_for_category(&self, category_name: &str) -> Vec<DictionaryEntry> {
        // For unknown dictionary, we need exact match
        self.inner
            .common_prefix_search(category_name)
            .into_iter()
            .filter(|e| e.length == category_name.len())
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
    /// # Arguments
    ///
    /// * `category` - The character category
    /// * `length` - The length of the unknown word surface in bytes
    pub fn generate_entries(
        &self,
        category: super::CharCategory,
        length: usize,
    ) -> Vec<DictionaryEntry> {
        let category_name = match category {
            super::CharCategory::Default => "DEFAULT",
            super::CharCategory::Space => "SPACE",
            super::CharCategory::Kanji => "KANJI",
            super::CharCategory::Symbol => "SYMBOL",
            super::CharCategory::Numeric => "NUMERIC",
            super::CharCategory::Alpha => "ALPHA",
            super::CharCategory::Hiragana => "HIRAGANA",
            super::CharCategory::Katakana => "KATAKANA",
            super::CharCategory::Kanjinumeric => "KANJINUMERIC",
            super::CharCategory::Greek => "GREEK",
            super::CharCategory::Cyrillic => "CYRILLIC",
        };

        // Get entries for this category and update their length
        self.get_entries_for_category(category_name)
            .into_iter()
            .map(|mut e| {
                e.length = length;
                e
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_unknown_dic_type() {
        assert_eq!(super::super::MECAB_UNK_DIC, 2);
    }
}
