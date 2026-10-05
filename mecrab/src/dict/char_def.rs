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

use super::sys_dic::DataBacking;

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
    /// Backing store (kept alive so map_ptr remains valid)
    _backing: DataBacking,
    /// Category names
    categories: Vec<String>,
    /// Pointer to CharInfo table (0xFFFF entries)
    map_ptr: *const CharInfo,
}

// SAFETY: `CharDef` only ever reads through `map_ptr`, never writes, and the
// memory it points at is the immutable backing store (`_backing`, an
// `Arc<Mmap>` or `Arc<Vec<u8>>`) that `self` keeps alive, so moving one
// between threads is sound.
unsafe impl Send for CharDef {}
// SAFETY: as the `Send` impl above — the only access through `map_ptr` is an
// immutable read of shared backing memory, so `&CharDef` is safe to share.
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

    /// Parse a byte slice and return `(categories, map_ptr)`.
    fn parse_bytes(data: &[u8]) -> Result<(Vec<String>, *const CharInfo)> {
        if data.len() < 4 {
            return Err(Error::CharDefError(
                "Character definition file too small".to_string(),
            ));
        }

        let csize = LittleEndian::read_u32(&data[0..4]) as usize;

        // Checked: on a 32-bit target (wasm32) `csize * 32` can wrap, which
        // would let a short file pass the length check below.
        let expected_size = Self::file_len(csize).ok_or_else(|| {
            Error::CharDefError(format!(
                "Character definition category count {} overflows the address space",
                csize
            ))
        })?;
        if data.len() != expected_size {
            return Err(Error::CharDefError(format!(
                "Character definition file size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

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

        let map_ptr = data[offset..].as_ptr() as *const CharInfo;
        // `get_char_info` reads whole `CharInfo`s (a `u32`, align 4) through
        // this pointer. The table starts at `4 + 32 * csize`, a multiple of 4,
        // so it is aligned in a memory map or allocator buffer; check it rather
        // than assume it.
        if !map_ptr.is_aligned() {
            return Err(Error::CharDefError(format!(
                "Character table at offset {} is not {}-byte aligned in memory",
                offset,
                std::mem::align_of::<CharInfo>()
            )));
        }
        Ok((categories, map_ptr))
    }

    /// The file length for `csize` categories (`4 + 32 * csize + TABLE_SIZE *
    /// CharInfo::SIZE`), or `None` when that does not fit in `usize`.
    fn file_len(csize: usize) -> Option<usize> {
        csize
            .checked_mul(32)?
            .checked_add(4)?
            .checked_add(Self::TABLE_SIZE * CharInfo::SIZE)
    }

    /// Load character definitions from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupted.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let (categories, map_ptr) = Self::parse_bytes(mmap.as_ref())?;
        Ok(Self {
            _backing: DataBacking::Mmap(mmap),
            categories,
            map_ptr,
        })
    }

    /// Load character definitions from an owned byte buffer (no filesystem required).
    ///
    /// # Errors
    ///
    /// Returns an error if the data is corrupted.
    pub fn from_bytes_owned(data: Arc<Vec<u8>>) -> Result<Self> {
        let (categories, map_ptr) = Self::parse_bytes(data.as_ref())?;
        Ok(Self {
            _backing: DataBacking::Owned(data),
            categories,
            map_ptr,
        })
    }

    /// Get CharInfo for a Unicode code point
    #[inline]
    pub fn get_char_info(&self, c: char) -> CharInfo {
        let code = c as u32;
        if code < Self::TABLE_SIZE as u32 {
            // SAFETY: `code < TABLE_SIZE` was just checked, and `map_ptr`
            // addresses `TABLE_SIZE` contiguous immutable `CharInfo`s:
            // `parse_bytes` checked the file length without overflow and
            // checked `map_ptr` is 4-byte aligned, and `_backing` keeps the
            // bytes alive for `self`, so the read is inside that allocation.
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
const CHAR_CACHE_SLOTS: usize = 256;

#[cfg(test)]
#[derive(Clone, Copy)]
struct CharCacheEntry {
    code_point: u32,
    #[allow(dead_code)]
    info: CharInfo,
    valid: bool,
}

#[cfg(test)]
impl Default for CharCacheEntry {
    fn default() -> Self {
        Self {
            code_point: 0,
            info: CharInfo::default(),
            valid: false,
        }
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
    fn test_file_len_is_checked() {
        let table = CharDef::TABLE_SIZE * CharInfo::SIZE;
        assert_eq!(CharDef::file_len(0), Some(4 + table));
        assert_eq!(CharDef::file_len(3), Some(4 + 96 + table));
        // `csize * 32` and each addition can overflow on its own.
        assert_eq!(CharDef::file_len(usize::MAX / 32 + 1), None);
        assert_eq!(CharDef::file_len(usize::MAX / 32), None);
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

    #[test]
    fn test_char_cache_slot_bounds() {
        // Verify slot assignment stays within bounds for representative Japanese ranges
        let slot_hiragana = ('あ' as u32) as usize & (CHAR_CACHE_SLOTS - 1);
        let slot_katakana = ('ア' as u32) as usize & (CHAR_CACHE_SLOTS - 1);
        let slot_kanji = ('漢' as u32) as usize & (CHAR_CACHE_SLOTS - 1);
        let slot_ascii = ('A' as u32) as usize & (CHAR_CACHE_SLOTS - 1);

        assert!(slot_hiragana < CHAR_CACHE_SLOTS);
        assert!(slot_katakana < CHAR_CACHE_SLOTS);
        assert!(slot_kanji < CHAR_CACHE_SLOTS);
        assert!(slot_ascii < CHAR_CACHE_SLOTS);
    }

    #[test]
    fn test_char_cache_entry_default() {
        let entry = CharCacheEntry::default();
        assert!(!entry.valid);
        assert_eq!(entry.code_point, 0);
    }

    #[test]
    fn test_char_cache_slots_power_of_two() {
        // CHAR_CACHE_SLOTS must be a power of two for the AND-mask trick
        assert_eq!(CHAR_CACHE_SLOTS & (CHAR_CACHE_SLOTS - 1), 0);
    }
}
