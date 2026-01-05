//! System dictionary (sys.dic) parser
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module handles loading the binary system dictionary file (sys.dic)
//! which contains the Double-Array Trie, tokens, and feature strings.
//!
//! Reference: ../ref/mecab-0.996/src/dictionary.cpp

use crate::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::sync::Arc;

use super::DictionaryEntry;
use super::double_array_trie::{DartsResult, DoubleArrayTrie};

/// Magic number for system dictionary validation
/// From MeCab: const unsigned int DictionaryMagicID = 0xef718f77u;
pub const DICTIONARY_MAGIC_ID: u32 = 0xef71_8f77;

/// Dictionary version (DIC_VERSION from MeCab)
pub const DIC_VERSION: u32 = 102;

/// Header size: 10 * u32 (40 bytes) + 32 bytes charset = 72 bytes
pub const HEADER_SIZE: usize = 72;

/// Maximum number of results from common prefix search
const MAX_RESULTS: usize = 512;

/// Token structure matching MeCab's Token struct (16 bytes)
/// Reference: ../ref/mecab-0.996/src/dictionary.h
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Token {
    /// Left context attribute (lcAttr)
    pub left_id: u16,
    /// Right context attribute (rcAttr)
    pub right_id: u16,
    /// Part-of-speech ID (posid)
    pub pos_id: u16,
    /// Word cost (wcost)
    pub wcost: i16,
    /// Offset to feature string
    pub feature_offset: u32,
    /// Compound info (unused in basic analysis)
    pub compound: u32,
}

impl Token {
    /// Size of Token in bytes
    pub const SIZE: usize = 16;
}

/// System dictionary containing trie, tokens, and features
pub struct SysDic {
    /// Memory map (kept alive)
    _mmap: Arc<Mmap>,
    /// Double-Array Trie
    trie: DoubleArrayTrie,
    /// Pointer to token array
    tokens_ptr: *const Token,
    /// Number of tokens
    tokens_count: usize,
    /// Pointer to feature strings
    features_ptr: *const u8,
    /// Size of feature section
    features_size: usize,
    /// Dictionary version
    version: u32,
    /// Dictionary type (0=sys, 1=usr, 2=unk)
    dict_type: u32,
    /// Lexicon size (number of entries)
    lexicon_size: u32,
    /// Left context size
    left_size: u32,
    /// Right context size
    right_size: u32,
    /// Character set (e.g., "UTF-8")
    charset: String,
}

// Safety: The pointers point to immutable memory-mapped data
unsafe impl Send for SysDic {}
unsafe impl Sync for SysDic {}

impl std::fmt::Debug for SysDic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SysDic")
            .field("version", &self.version)
            .field("dict_type", &self.dict_type)
            .field("lexicon_size", &self.lexicon_size)
            .field("left_size", &self.left_size)
            .field("right_size", &self.right_size)
            .field("charset", &self.charset)
            .field("trie_size", &self.trie.size())
            .field("tokens_count", &self.tokens_count)
            .finish()
    }
}

impl SysDic {
    /// Load system dictionary from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupted or has invalid format.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let data = &mmap[..];

        if data.len() < HEADER_SIZE {
            return Err(Error::CorruptedDictionary(format!(
                "Dictionary file too small: {} bytes (minimum {} bytes)",
                data.len(),
                HEADER_SIZE
            )));
        }

        // Read and validate magic number
        // magic ^ DictionaryMagicID == filesize
        let magic = LittleEndian::read_u32(&data[0..4]);
        let expected_size = magic ^ DICTIONARY_MAGIC_ID;

        if expected_size != data.len() as u32 {
            return Err(Error::InvalidDictionaryFormat(format!(
                "Magic number mismatch: expected file size {}, got {}",
                expected_size,
                data.len()
            )));
        }

        // Read header fields
        let version = LittleEndian::read_u32(&data[4..8]);
        if version != DIC_VERSION {
            return Err(Error::InvalidDictionaryFormat(format!(
                "Incompatible dictionary version: expected {}, got {}",
                DIC_VERSION, version
            )));
        }

        let dict_type = LittleEndian::read_u32(&data[8..12]);
        let lexicon_size = LittleEndian::read_u32(&data[12..16]);
        let left_size = LittleEndian::read_u32(&data[16..20]);
        let right_size = LittleEndian::read_u32(&data[20..24]);
        let da_size = LittleEndian::read_u32(&data[24..28]) as usize;
        let token_size = LittleEndian::read_u32(&data[28..32]) as usize;
        let feature_size = LittleEndian::read_u32(&data[32..36]) as usize;
        // data[36..40] is dummy/padding

        // Read charset (32 bytes, null-terminated)
        let charset_bytes = &data[40..72];
        let charset_end = charset_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(charset_bytes.len());
        let charset = String::from_utf8_lossy(&charset_bytes[..charset_end]).to_string();

        // Validate sizes
        let expected_total = HEADER_SIZE + da_size + token_size + feature_size;
        if data.len() < expected_total {
            return Err(Error::CorruptedDictionary(format!(
                "Dictionary file truncated: expected {} bytes, got {}",
                expected_total,
                data.len()
            )));
        }

        // Create Double-Array Trie
        let da_offset = HEADER_SIZE;
        let trie = DoubleArrayTrie::from_bytes(&data[da_offset..], da_size)?;

        // Get token array pointer
        let token_offset = da_offset + da_size;
        let tokens_ptr = data[token_offset..].as_ptr() as *const Token;
        let tokens_count = token_size / Token::SIZE;

        // Get feature strings pointer
        let feature_offset = token_offset + token_size;
        let features_ptr = data[feature_offset..].as_ptr();
        let features_size = feature_size;

        Ok(Self {
            _mmap: mmap,
            trie,
            tokens_ptr,
            tokens_count,
            features_ptr,
            features_size,
            version,
            dict_type,
            lexicon_size,
            left_size,
            right_size,
            charset,
        })
    }

    /// Perform common prefix search and return dictionary entries
    pub fn common_prefix_search(&self, key: &str) -> Vec<DictionaryEntry> {
        let key_bytes = key.as_bytes();
        let mut results = [DartsResult::default(); MAX_RESULTS];

        let num_results = self.trie.common_prefix_search(key_bytes, &mut results);

        let mut entries = Vec::new();

        for result in results.iter().take(num_results) {
            // Decode the value:
            // - Upper bits (value >> 8): token start index
            // - Lower bits (value & 0xff): token count
            let value = result.value as u32;
            let token_start = (value >> 8) as usize;
            let token_count = (value & 0xff) as usize;

            for i in 0..token_count {
                let token_idx = token_start + i;
                if let Some(token) = self.get_token(token_idx) {
                    let feature = self.get_feature(token).to_string();

                    entries.push(DictionaryEntry {
                        length: result.length,
                        word_id: token_idx as u32,
                        left_id: token.left_id,
                        right_id: token.right_id,
                        pos_id: token.pos_id,
                        wcost: token.wcost,
                        feature,
                    });
                }
            }
        }

        entries
    }

    /// Get a token by index
    #[inline]
    fn get_token(&self, index: usize) -> Option<&Token> {
        if index < self.tokens_count {
            // Safety: We verified the index is in bounds
            Some(unsafe { &*self.tokens_ptr.add(index) })
        } else {
            None
        }
    }

    /// Get feature string for a token
    pub fn get_feature(&self, token: &Token) -> &str {
        let offset = token.feature_offset as usize;
        if offset >= self.features_size {
            return "";
        }

        // Safety: We verified the offset is in bounds
        let ptr = unsafe { self.features_ptr.add(offset) };

        // Find null terminator
        let mut len = 0;
        while len < self.features_size - offset {
            if unsafe { *ptr.add(len) } == 0 {
                break;
            }
            len += 1;
        }

        // Safety: We found the null terminator
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        std::str::from_utf8(slice).unwrap_or("")
    }

    /// Get the charset
    pub fn charset(&self) -> &str {
        &self.charset
    }

    /// Get the lexicon size
    pub fn lexicon_size(&self) -> usize {
        self.lexicon_size as usize
    }

    /// Get the dictionary version
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Get the total number of tokens
    pub fn token_count(&self) -> usize {
        self.tokens_count
    }

    /// Get a token by index (public version)
    pub fn token_at(&self, index: usize) -> Option<&Token> {
        self.get_token(index)
    }

    /// Get the dictionary type
    pub fn dict_type(&self) -> u32 {
        self.dict_type
    }

    /// Get the left context size
    pub fn left_size(&self) -> usize {
        self.left_size as usize
    }

    /// Get the right context size
    pub fn right_size(&self) -> usize {
        self.right_size as usize
    }

    /// Get the trie size in units
    pub fn trie_size(&self) -> usize {
        self.trie.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_size() {
        assert_eq!(std::mem::size_of::<Token>(), 16);
        assert_eq!(Token::SIZE, 16);
    }

    #[test]
    fn test_magic_id() {
        assert_eq!(DICTIONARY_MAGIC_ID, 0xef71_8f77);
    }

    #[test]
    fn test_header_size() {
        assert_eq!(HEADER_SIZE, 72);
    }
}
