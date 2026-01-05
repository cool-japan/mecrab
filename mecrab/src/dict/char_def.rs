//! Character category definitions for unknown word handling
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module defines character categories used for handling unknown
//! (out-of-vocabulary) words in the morphological analysis.
//!
//! Reference: ../ref/mecab-0.996/src/char_property.cpp
//!
//! Binary format (char.bin):
//! - First 4 bytes: csize (u32) - number of categories
//! - Next csize * 32 bytes: category names (32 bytes each, null-padded)
//! - Next 0xffff * 4 bytes: CharInfo table indexed by UCS-2 code point

use crate::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::sync::Arc;

/// Character category names (matching IPADIC)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum CharCategory {
    /// Default category
    #[default]
    Default = 0,
    /// Space/whitespace
    Space = 1,
    /// Kanji (CJK ideographs)
    Kanji = 2,
    /// Symbols
    Symbol = 3,
    /// Numeric/digits
    Numeric = 4,
    /// Alphabetic (ASCII letters)
    Alpha = 5,
    /// Hiragana
    Hiragana = 6,
    /// Katakana
    Katakana = 7,
    /// Kanjinumeric (kanji numbers)
    Kanjinumeric = 8,
    /// Greek letters
    Greek = 9,
    /// Cyrillic letters
    Cyrillic = 10,
}

impl From<u8> for CharCategory {
    fn from(value: u8) -> Self {
        match value {
            1 => Self::Space,
            2 => Self::Kanji,
            3 => Self::Symbol,
            4 => Self::Numeric,
            5 => Self::Alpha,
            6 => Self::Hiragana,
            7 => Self::Katakana,
            8 => Self::Kanjinumeric,
            9 => Self::Greek,
            10 => Self::Cyrillic,
            _ => Self::Default,
        }
    }
}

/// Character information packed in 4 bytes
/// Bit layout:
/// - type:         18 bits (bitmask of category types)
/// - default_type:  8 bits (default category ID)
/// - length:        4 bits (max length for grouping)
/// - group:         1 bit  (whether to group consecutive chars)
/// - invoke:        1 bit  (whether to invoke unknown word processing)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct CharInfo {
    /// Raw packed value
    packed: u32,
}

impl CharInfo {
    /// Size of CharInfo in bytes
    pub const SIZE: usize = 4;

    /// Get the type bitmask (18 bits)
    #[inline]
    pub fn type_mask(&self) -> u32 {
        self.packed & 0x3FFFF // 18 bits
    }

    /// Get the default type (8 bits)
    #[inline]
    pub fn default_type(&self) -> u8 {
        ((self.packed >> 18) & 0xFF) as u8
    }

    /// Get the max length for grouping (4 bits)
    #[inline]
    pub fn length(&self) -> u8 {
        ((self.packed >> 26) & 0xF) as u8
    }

    /// Check if grouping is enabled (1 bit)
    #[inline]
    pub fn group(&self) -> bool {
        ((self.packed >> 30) & 1) != 0
    }

    /// Check if unknown word processing should be invoked (1 bit)
    #[inline]
    pub fn invoke(&self) -> bool {
        ((self.packed >> 31) & 1) != 0
    }

    /// Get the default category
    #[inline]
    pub fn category(&self) -> CharCategory {
        CharCategory::from(self.default_type())
    }

    /// Check if this CharInfo is of a specific type
    #[inline]
    pub fn is_kind_of(&self, other: CharInfo) -> bool {
        (self.type_mask() & other.type_mask()) != 0
    }
}

/// Character definition table
pub struct CharDef {
    /// Memory map (kept alive)
    _mmap: Arc<Mmap>,
    /// Category names
    categories: Vec<String>,
    /// Pointer to CharInfo table (0xFFFF entries)
    map_ptr: *const CharInfo,
}

// Safety: The map_ptr points to immutable memory-mapped data
unsafe impl Send for CharDef {}
unsafe impl Sync for CharDef {}

impl std::fmt::Debug for CharDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CharDef")
            .field("categories", &self.categories)
            .finish()
    }
}

impl CharDef {
    /// Number of entries in the CharInfo table (0xFFFF = 65535)
    pub const TABLE_SIZE: usize = 0xFFFF;

    /// Load character definitions from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupted.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let data = &mmap[..];

        if data.len() < 4 {
            return Err(Error::CharDefError(
                "Character definition file too small".to_string(),
            ));
        }

        // Read the number of categories
        let csize = LittleEndian::read_u32(&data[0..4]) as usize;

        // Validate file size
        // Expected: 4 + (csize * 32) + (0xFFFF * 4)
        let expected_size = 4 + (csize * 32) + (Self::TABLE_SIZE * CharInfo::SIZE);
        if data.len() != expected_size {
            return Err(Error::CharDefError(format!(
                "Character definition file size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        // Read category names
        let mut categories = Vec::with_capacity(csize);
        let mut offset = 4;
        for _ in 0..csize {
            let name_bytes = &data[offset..offset + 32];
            let name_end = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            let name = String::from_utf8_lossy(&name_bytes[..name_end]).to_string();
            categories.push(name);
            offset += 32;
        }

        // Get pointer to CharInfo table
        let map_ptr = data[offset..].as_ptr() as *const CharInfo;

        Ok(Self {
            _mmap: mmap,
            categories,
            map_ptr,
        })
    }

    /// Get CharInfo for a Unicode code point
    #[inline]
    pub fn get_char_info(&self, c: char) -> CharInfo {
        let code = c as u32;
        if code < Self::TABLE_SIZE as u32 {
            // Safety: We verified the code is in bounds
            unsafe { *self.map_ptr.add(code as usize) }
        } else {
            // For characters outside BMP, return default
            CharInfo::default()
        }
    }

    /// Get CharInfo for a byte sequence (handles UTF-8)
    pub fn get_char_info_from_bytes(&self, bytes: &[u8]) -> (CharInfo, usize) {
        if bytes.is_empty() {
            return (CharInfo::default(), 0);
        }

        // Decode UTF-8 to get the first character
        let s = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return (CharInfo::default(), 1),
        };

        if let Some(c) = s.chars().next() {
            let len = c.len_utf8();
            (self.get_char_info(c), len)
        } else {
            (CharInfo::default(), 0)
        }
    }

    /// Get category name by ID
    pub fn category_name(&self, id: usize) -> Option<&str> {
        self.categories.get(id).map(String::as_str)
    }

    /// Get number of categories
    pub fn category_count(&self) -> usize {
        self.categories.len()
    }

    /// Get category ID by name
    pub fn category_id(&self, name: &str) -> Option<usize> {
        self.categories.iter().position(|n| n == name)
    }

    /// Check if a category should group consecutive characters
    ///
    /// This uses the group flag from the first character of the category.
    pub fn should_group(&self, category: CharCategory) -> bool {
        // Use a representative character for each category to check grouping
        let sample_char = match category {
            CharCategory::Default => ' ',
            CharCategory::Space => ' ',
            CharCategory::Kanji => '漢',
            CharCategory::Symbol => '!',
            CharCategory::Numeric => '0',
            CharCategory::Alpha => 'A',
            CharCategory::Hiragana => 'あ',
            CharCategory::Katakana => 'ア',
            CharCategory::Kanjinumeric => '一',
            CharCategory::Greek => 'Α',
            CharCategory::Cyrillic => 'А',
        };

        self.get_char_info(sample_char).group()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charinfo_size() {
        assert_eq!(std::mem::size_of::<CharInfo>(), 4);
        assert_eq!(CharInfo::SIZE, 4);
    }

    #[test]
    fn test_charinfo_default() {
        let info = CharInfo::default();
        assert_eq!(info.type_mask(), 0);
        assert_eq!(info.default_type(), 0);
        assert_eq!(info.length(), 0);
        assert!(!info.group());
        assert!(!info.invoke());
    }

    #[test]
    fn test_char_category() {
        assert_eq!(CharCategory::from(0), CharCategory::Default);
        assert_eq!(CharCategory::from(1), CharCategory::Space);
        assert_eq!(CharCategory::from(6), CharCategory::Hiragana);
        assert_eq!(CharCategory::from(7), CharCategory::Katakana);
        assert_eq!(CharCategory::from(255), CharCategory::Default);
    }
}
