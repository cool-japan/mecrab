//! Feature string table for dictionary entries
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module handles the feature string table that maps feature IDs
//! to human-readable feature strings (POS, reading, pronunciation, etc.).

use crate::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};

/// Feature table storing feature strings
#[derive(Debug)]
pub struct FeatureTable {
    /// Feature strings indexed by feature ID
    features: Vec<String>,
}

impl FeatureTable {
    /// Create a feature table from raw dictionary bytes
    ///
    /// The feature data is located after the trie data in sys.dic.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is corrupted.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // The feature section starts after the trie data
        // For now, we'll extract what we can from the dictionary

        // Skip the header and trie arrays to find feature data
        if data.len() < 72 {
            return Err(Error::FeatureParseError(
                "Data too small for feature table".to_string(),
            ));
        }

        let feature_size = LittleEndian::read_u32(&data[32..36]) as usize;

        // For now, return an empty table - actual parsing requires
        // understanding the exact dictionary format
        let features = Vec::with_capacity(feature_size);

        Ok(Self { features })
    }

    /// Get a feature string by ID
    pub fn get(&self, feature_id: u32) -> Option<&str> {
        self.features.get(feature_id as usize).map(String::as_str)
    }

    /// Get the number of features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if the table is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Parse a feature string into components
    ///
    /// IPADIC features are comma-separated:
    /// `品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,基本形,読み,発音`
    pub fn parse_feature(feature: &str) -> FeatureComponents<'_> {
        let parts: Vec<&str> = feature.split(',').collect();

        FeatureComponents {
            pos1: parts.first().copied(),
            pos2: parts.get(1).copied(),
            pos3: parts.get(2).copied(),
            pos4: parts.get(3).copied(),
            conjugation_type: parts.get(4).copied(),
            conjugation_form: parts.get(5).copied(),
            base_form: parts.get(6).copied(),
            reading: parts.get(7).copied(),
            pronunciation: parts.get(8).copied(),
        }
    }
}

/// Parsed feature components
#[derive(Debug, Clone)]
pub struct FeatureComponents<'a> {
    /// Part of speech level 1
    pub pos1: Option<&'a str>,
    /// Part of speech level 2
    pub pos2: Option<&'a str>,
    /// Part of speech level 3
    pub pos3: Option<&'a str>,
    /// Part of speech level 4
    pub pos4: Option<&'a str>,
    /// Conjugation type
    pub conjugation_type: Option<&'a str>,
    /// Conjugation form
    pub conjugation_form: Option<&'a str>,
    /// Base/dictionary form
    pub base_form: Option<&'a str>,
    /// Reading (katakana)
    pub reading: Option<&'a str>,
    /// Pronunciation
    pub pronunciation: Option<&'a str>,
}

impl FeatureComponents<'_> {
    /// Get the full POS string (pos1-pos2-pos3-pos4)
    pub fn full_pos(&self) -> String {
        [self.pos1, self.pos2, self.pos3, self.pos4]
            .iter()
            .filter_map(|&p| p.filter(|&s| s != "*"))
            .collect::<Vec<_>>()
            .join("-")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_feature() {
        let feature = "名詞,一般,*,*,*,*,東京,トウキョウ,トーキョー";
        let components = FeatureTable::parse_feature(feature);

        assert_eq!(components.pos1, Some("名詞"));
        assert_eq!(components.pos2, Some("一般"));
        assert_eq!(components.base_form, Some("東京"));
        assert_eq!(components.reading, Some("トウキョウ"));
        assert_eq!(components.pronunciation, Some("トーキョー"));
    }

    #[test]
    fn test_full_pos() {
        let feature = "動詞,自立,*,*,五段・カ行イ音便,連用タ接続,書く,カイ,カイ";
        let components = FeatureTable::parse_feature(feature);

        assert_eq!(components.full_pos(), "動詞-自立");
    }
}
