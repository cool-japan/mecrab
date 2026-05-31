//! Dictionary module for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module handles loading and managing MeCab-compatible dictionaries.
//! It supports IPADIC format with memory-mapped file I/O for efficiency.
//!
//! Reference: ../ref/mecab-0.996/src/dictionary.cpp

mod char_def;
mod connection_matrix;
mod double_array_trie;
mod feature;
mod overlay;
pub mod provider;
mod sys_dic;
mod unknown;
pub mod user_dict;

pub use char_def::{CharCategory, CharDef, CharInfo};
pub use connection_matrix::ConnectionMatrix;
pub use double_array_trie::{DartsResult, DoubleArrayTrie};
pub use feature::FeatureTable;
pub use overlay::{OverlayDictionary, OverlayEntry};
pub use provider::{
    AutoDetectProvider, DictionaryFormat, DictionaryProvider, IpadicProvider, MorphemeFeatures,
    NeologdProvider, UnidicProvider,
};
pub use sys_dic::{SysDic, Token};
pub use unknown::UnknownDictionary;
pub use user_dict::{DictFormat, UserDictManager, UserDictStats, UserEntry, ValidationResult};

use crate::{Error, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// Dictionary file names (MeCab/IPADIC format)
/// System dictionary file name
pub const SYS_DIC_FILE: &str = "sys.dic";
/// Unknown word dictionary file name
pub const UNK_DIC_FILE: &str = "unk.dic";
/// Connection matrix file name
pub const MATRIX_FILE: &str = "matrix.bin";
/// Character property binary file name
pub const CHAR_BIN_FILE: &str = "char.bin";

/// Dictionary type constants (from MeCab)
/// System dictionary type
pub const MECAB_SYS_DIC: u32 = 0;
/// User dictionary type
pub const MECAB_USR_DIC: u32 = 1;
/// Unknown word dictionary type
pub const MECAB_UNK_DIC: u32 = 2;

/// Type alias for surface form → URIs mapping
pub type SurfaceMap = std::collections::HashMap<String, Vec<(String, f32)>>;

/// The main dictionary structure containing all loaded dictionary data
pub struct Dictionary {
    /// System dictionary (contains trie, tokens, features)
    pub sys_dic: SysDic,
    /// Unknown word dictionary
    pub unknown: UnknownDictionary,
    /// Connection matrix for transition costs
    pub matrix: ConnectionMatrix,
    /// Character category definitions
    pub char_def: CharDef,
    /// Overlay dictionary for runtime word additions
    pub overlay: OverlayDictionary,
    /// Semantic pool for entity URIs (optional)
    pub semantic_pool: Option<Arc<crate::semantic::pool::SemanticPool>>,
    /// Surface form → URIs mapping (optional)
    pub surface_map: Option<Arc<SurfaceMap>>,
    /// Trained word-cost overrides: word_id → delta to apply on top of sys_dic wcost.
    ///
    /// Protected by `RwLock` for thread-safe updates.  The lock is only
    /// acquired when `override_count > 0` (fast-path: atomic check first).
    word_cost_overrides: RwLock<HashMap<u32, i16>>,
    /// Number of active word-cost overrides. 0 → skip the slow path entirely.
    ///
    /// Set with `Ordering::Release` on write; loaded with `Ordering::Acquire` on read.
    override_count: AtomicUsize,
    /// Memory maps (kept alive to maintain the mapped regions)
    _mmaps: Vec<Arc<Mmap>>,
}

impl std::fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary")
            .field("sys_dic", &self.sys_dic)
            .field("matrix", &self.matrix)
            .field("char_def", &self.char_def)
            .field("overlay", &self.overlay)
            .finish()
    }
}

impl Dictionary {
    /// Load a dictionary from the specified directory
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dictionary directory containing sys.dic, matrix.bin, etc.
    ///
    /// # Errors
    ///
    /// Returns an error if any dictionary file is missing or corrupted.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(Error::DictionaryNotFound(path.to_path_buf()));
        }

        let mut mmaps = Vec::new();

        // Load system dictionary
        let sys_path = path.join(SYS_DIC_FILE);
        let sys_mmap = Arc::new(Self::open_mmap(&sys_path)?);
        let sys_dic = SysDic::from_mmap(Arc::clone(&sys_mmap))?;
        mmaps.push(sys_mmap);

        // Load connection matrix
        let matrix_path = path.join(MATRIX_FILE);
        let matrix_mmap = Arc::new(Self::open_mmap(&matrix_path)?);
        let matrix = ConnectionMatrix::from_mmap(Arc::clone(&matrix_mmap))?;
        mmaps.push(matrix_mmap);

        // Load character definitions
        let char_path = path.join(CHAR_BIN_FILE);
        let char_mmap = Arc::new(Self::open_mmap(&char_path)?);
        let char_def = CharDef::from_mmap(Arc::clone(&char_mmap))?;
        mmaps.push(char_mmap);

        // Load unknown word dictionary
        let unk_path = path.join(UNK_DIC_FILE);
        let unk_mmap = Arc::new(Self::open_mmap(&unk_path)?);
        let unknown = UnknownDictionary::from_mmap(Arc::clone(&unk_mmap))?;
        mmaps.push(unk_mmap);

        Ok(Self {
            sys_dic,
            unknown,
            matrix,
            char_def,
            overlay: OverlayDictionary::new(),
            semantic_pool: None,
            surface_map: None,
            word_cost_overrides: RwLock::new(HashMap::new()),
            override_count: AtomicUsize::new(0),
            _mmaps: mmaps,
        })
    }

    /// Load a dictionary with semantic pool from the specified directory
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the dictionary directory
    /// * `semantic_path` - Path to the semantic pool file (semantic.bin)
    ///
    /// # Errors
    ///
    /// Returns an error if any dictionary file is missing or corrupted.
    pub fn load_with_semantics(path: &Path, semantic_path: &Path) -> Result<Self> {
        let mut dict = Self::load(path)?;

        // Load semantic pool
        if semantic_path.exists() {
            let pool_file = File::open(semantic_path)?;
            let pool_data = unsafe { Mmap::map(&pool_file)? };
            let pool = crate::semantic::pool::SemanticPool::from_bytes(&pool_data)?;
            dict.semantic_pool = Some(Arc::new(pool));

            // Try to load surface map from the same directory
            let semantic_dir = semantic_path.parent().unwrap_or(Path::new("."));
            let map_path = semantic_dir.join("surface_map.json");
            if map_path.exists() {
                let map_data = std::fs::read_to_string(&map_path)?;
                let map: SurfaceMap = serde_json::from_str(&map_data)?;
                dict.surface_map = Some(Arc::new(map));
            }
        }

        Ok(dict)
    }

    /// Open a file and create a memory map
    fn open_mmap(path: &Path) -> Result<Mmap> {
        if !path.exists() {
            return Err(Error::DictionaryNotFound(path.to_path_buf()));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(mmap)
    }

    /// Load a dictionary from raw byte slices (no filesystem access required).
    ///
    /// This is the primary entry point for WASM / in-memory dictionary loading.
    /// Each slice must contain the exact binary content of the corresponding file:
    ///
    /// | Parameter  | Corresponding file |
    /// |------------|--------------------|
    /// | `sys_dic`  | `sys.dic`          |
    /// | `matrix`   | `matrix.bin`       |
    /// | `char_def` | `char.bin`         |
    /// | `unk_def`  | `unk.dic`          |
    ///
    /// The slices are copied into owned `Arc<Vec<u8>>` buffers so the caller
    /// does not need to keep them alive.
    ///
    /// # Errors
    ///
    /// Returns an error if any component is corrupted or has an invalid format.
    pub fn from_bytes(
        sys_dic: &[u8],
        matrix: &[u8],
        char_def: &[u8],
        unk_def: &[u8],
    ) -> Result<Self> {
        let sys_dic = SysDic::from_bytes_owned(Arc::new(sys_dic.to_vec()))?;
        let matrix = ConnectionMatrix::from_bytes_owned(Arc::new(matrix.to_vec()))?;
        let char_def = CharDef::from_bytes_owned(Arc::new(char_def.to_vec()))?;
        let unknown = UnknownDictionary::from_bytes_owned(Arc::new(unk_def.to_vec()))?;

        Ok(Self {
            sys_dic,
            unknown,
            matrix,
            char_def,
            overlay: OverlayDictionary::new(),
            semantic_pool: None,
            surface_map: None,
            word_cost_overrides: RwLock::new(HashMap::new()),
            override_count: AtomicUsize::new(0),
            _mmaps: Vec::new(),
        })
    }

    /// Try to load the default dictionary from standard locations
    ///
    /// # Errors
    ///
    /// Returns an error if no dictionary is found in any standard location.
    pub fn default_dictionary() -> Result<Self> {
        // Standard dictionary locations (ordered by preference)
        let locations = [
            "/var/lib/mecab/dic/ipadic-utf8",
            "/usr/lib/mecab/dic/ipadic-utf8",
            "/usr/local/lib/mecab/dic/ipadic-utf8",
            "/usr/share/mecab/dic/ipadic-utf8",
            "/usr/lib/mecab/dic/ipadic",
            "/usr/local/lib/mecab/dic/ipadic",
            "/usr/share/mecab/dic/ipadic",
            "/usr/lib64/mecab/dic/ipadic",
        ];

        // Also check home directory
        if let Some(home) = std::env::var_os("HOME") {
            let home_path = Path::new(&home).join(".local/share/mecrab/dic/ipadic");
            if home_path.exists() {
                return Self::load(&home_path);
            }
        }

        for location in &locations {
            let path = Path::new(location);
            if path.exists() {
                return Self::load(path);
            }
        }

        Err(Error::DefaultDictionaryNotFound)
    }

    /// Look up a word in the dictionary using common prefix search
    ///
    /// This checks the overlay dictionary first, then the system dictionary.
    /// Returns all matching entries from both layers.
    ///
    /// ## Fast path
    ///
    /// When no words have been added to the overlay **and** no word-cost
    /// overrides are active (the common case in production), this method
    /// bypasses all RwLock operations and returns system-dictionary results
    /// directly.  Both checks use a single `AtomicUsize::load(Acquire)` each
    /// — no mutex or fence beyond that.
    ///
    /// ## Invariant
    ///
    /// When `override_count == 0 && overlay.is_empty()`, lookup returns
    /// **identical** results to the pre-change code path.
    pub fn lookup(&self, key: &str) -> Vec<DictionaryEntry> {
        let has_overrides = self.override_count.load(Ordering::Acquire) > 0;

        // Ultra-fast path: skip both overlay and overrides.
        if self.overlay.is_empty() && !has_overrides {
            return self.sys_dic.common_prefix_search(key);
        }

        // Collect base results (overlay + sys_dic)
        let mut results = if self.overlay.is_empty() {
            self.sys_dic.common_prefix_search(key)
        } else {
            let mut r = self.overlay.lookup(key);
            r.extend(self.sys_dic.common_prefix_search(key));
            r
        };

        // Apply word-cost overrides when active
        if has_overrides {
            let overrides = self.word_cost_overrides.read().unwrap_or_else(|e| e.into_inner());
            for entry in &mut results {
                if let Some(&delta) = overrides.get(&entry.word_id) {
                    let updated = entry.wcost as i64 + delta as i64;
                    entry.wcost = updated.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
                }
            }
        }

        results
    }

    /// Set word-cost overrides from trained deltas. Thread-safe.
    ///
    /// Each entry in `overrides` is a `word_id → delta_i16` pair.  The delta
    /// is applied on top of the word's original `wcost` from the system
    /// dictionary on every subsequent call to [`lookup`](Self::lookup).
    ///
    /// Passing an empty map clears all overrides and re-enables the fast path.
    pub fn set_word_cost_overrides(&self, overrides: HashMap<u32, i16>) {
        let len = overrides.len();
        *self.word_cost_overrides.write().unwrap_or_else(|e| e.into_inner()) = overrides;
        self.override_count.store(len, Ordering::Release);
    }

    /// Load word-cost overrides from a TSV file (`word_id TAB delta_i16` per line).
    ///
    /// Lines beginning with `#` are treated as comments and skipped.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or any line is malformed.
    pub fn load_word_cost_overrides<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::IoError(format!("cannot open {}: {e}", path.as_ref().display())))?;
        let reader = BufReader::new(file);
        let mut map: HashMap<u32, i16> = HashMap::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| Error::IoError(format!("{e}")))?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let mut parts = trimmed.splitn(2, '\t');
            let wid_str = parts.next().ok_or_else(|| {
                Error::IoError(format!("line {}: missing word_id", line_no + 1))
            })?;
            let delta_str = parts.next().ok_or_else(|| {
                Error::IoError(format!("line {}: missing delta", line_no + 1))
            })?;
            let word_id: u32 = wid_str.parse().map_err(|e| {
                Error::IoError(format!("line {}: invalid word_id: {e}", line_no + 1))
            })?;
            let delta: i16 = delta_str.parse().map_err(|e| {
                Error::IoError(format!("line {}: invalid delta: {e}", line_no + 1))
            })?;
            map.insert(word_id, delta);
        }
        self.set_word_cost_overrides(map);
        Ok(())
    }

    /// Add a word to the overlay dictionary at runtime
    ///
    /// This allows adding new words without restarting or reloading
    /// the main dictionary.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form (the actual text)
    /// * `entry` - The dictionary entry data
    ///
    /// # Example
    ///
    /// ```no_run
    /// // Requires a real MeCab dictionary installed
    /// // (install ipadic: `sudo apt install mecab-ipadic-utf8`)
    /// use mecrab::dict::{Dictionary, OverlayEntry};
    /// use std::path::Path;
    ///
    /// let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
    /// let dict = Dictionary::load(path).unwrap();
    /// dict.add_word("ChatGPT", OverlayEntry::new(
    ///     "名詞,固有名詞,一般,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー",
    ///     5000,
    /// ));
    /// ```
    pub fn add_word(&self, surface: &str, entry: OverlayEntry) {
        self.overlay.add_word(surface, entry);
    }

    /// Add a word with simple parameters
    ///
    /// Convenience method that creates a common noun entry.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form
    /// * `reading` - The katakana reading
    /// * `pronunciation` - The pronunciation
    /// * `wcost` - Word cost (lower = more preferred)
    pub fn add_simple_word(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        self.overlay
            .add_simple(surface, reading, pronunciation, wcost);
    }

    /// Remove a word from the overlay dictionary
    ///
    /// Note: This only removes words from the overlay layer.
    /// System dictionary entries cannot be removed.
    ///
    /// Returns true if the word was found and removed.
    pub fn remove_word(&self, surface: &str) -> bool {
        self.overlay.remove_word(surface)
    }

    /// Get the number of words in the overlay dictionary
    pub fn overlay_size(&self) -> usize {
        self.overlay.len()
    }

    /// Get the connection cost between two context IDs
    ///
    /// # Arguments
    ///
    /// * `right_id` - Right context ID of the left node
    /// * `left_id` - Left context ID of the right node
    #[inline]
    pub fn connection_cost(&self, right_id: u16, left_id: u16) -> i16 {
        self.matrix.cost(right_id, left_id)
    }

    /// Get connection cost without bounds checking (hot path optimization)
    ///
    /// # Safety
    ///
    /// Context IDs must be from valid dictionary entries.
    #[inline]
    pub unsafe fn connection_cost_unchecked(&self, right_id: u16, left_id: u16) -> i16 {
        // Safety: caller guarantees IDs are valid
        unsafe { self.matrix.cost_unchecked(right_id, left_id) }
    }

    /// Get the feature string for a token
    pub fn get_feature(&self, token: &Token) -> &str {
        self.sys_dic.get_feature(token)
    }

    /// Get character info for a character
    pub fn char_info(&self, c: char) -> CharInfo {
        self.char_def.get_char_info(c)
    }

    /// Get the character category for a character
    pub fn char_category(&self, c: char) -> CharCategory {
        self.char_def.get_char_info(c).category()
    }

    /// Get the charset of the dictionary
    pub fn charset(&self) -> &str {
        self.sys_dic.charset()
    }

    /// Get the number of entries in the dictionary
    pub fn size(&self) -> usize {
        self.sys_dic.lexicon_size()
    }

    /// Sample the number of comma-separated feature fields from the first available token.
    ///
    /// Returns `None` if the dictionary contains no tokens or the feature string is empty.
    /// This is used by the auto-detection pipeline to choose the correct
    /// `DictionaryProvider` without requiring the caller to know the format in advance.
    pub fn sample_feature_count(&self) -> Option<usize> {
        let token_count = self.sys_dic.token_count();
        for idx in 0..token_count {
            if let Some(token) = self.sys_dic.token_at(idx) {
                let feature = self.sys_dic.get_feature(token);
                if !feature.is_empty() {
                    let count = feature.split(',').count();
                    return Some(count);
                }
            }
        }
        None
    }
}

/// A dictionary entry returned from lookup
#[derive(Debug, Clone)]
pub struct DictionaryEntry {
    /// Length of the matched surface in bytes
    pub length: usize,
    /// Word ID (token index in dictionary, used for word embeddings)
    pub word_id: u32,
    /// Left context ID (for connection matrix)
    pub left_id: u16,
    /// Right context ID (for connection matrix)
    pub right_id: u16,
    /// Part of speech ID
    pub pos_id: u16,
    /// Word cost
    pub wcost: i16,
    /// Feature string (shared via Arc to avoid redundant allocations)
    pub feature: Arc<str>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_file_names() {
        assert_eq!(SYS_DIC_FILE, "sys.dic");
        assert_eq!(MATRIX_FILE, "matrix.bin");
        assert_eq!(CHAR_BIN_FILE, "char.bin");
        assert_eq!(UNK_DIC_FILE, "unk.dic");
    }

    #[test]
    fn test_dictionary_types() {
        assert_eq!(MECAB_SYS_DIC, 0);
        assert_eq!(MECAB_USR_DIC, 1);
        assert_eq!(MECAB_UNK_DIC, 2);
    }

    /// Verify `OverlayDictionary::is_empty()` starts true, becomes false after
    /// `add_word`, and returns true again after the word is removed.
    /// This is the standalone unit test for the overlay fast-path flag.
    #[test]
    fn test_overlay_is_empty_lifecycle() {
        let overlay = OverlayDictionary::new();
        assert!(overlay.is_empty(), "fresh overlay must be empty");

        overlay.add_word(
            "ChatGPT",
            OverlayEntry::new(
                "名詞,固有名詞,一般,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー",
                5000,
            ),
        );
        assert!(!overlay.is_empty(), "overlay must not be empty after add");

        let removed = overlay.remove_word("ChatGPT");
        assert!(removed, "remove_word should return true");
        assert!(overlay.is_empty(), "overlay must be empty after remove");
    }

    /// Verify that empty override map + empty overlay produces the identical
    /// fast-path result as having no overrides at all.
    ///
    /// Uses a synthetic in-memory dictionary so no real IPADIC is required.
    #[test]
    fn test_empty_override_lookup_identical() {
        use mecrab_builder::build_synthetic_dictionary;
        let sd = build_synthetic_dictionary();
        let dict = Dictionary::from_bytes(&sd.sys_dic, &sd.matrix, &sd.char_def, &sd.unk_def)
            .expect("build dict");

        // No overrides set — fast path
        let base = dict.lookup("すもも");

        // Set and then clear overrides — must return to fast path
        dict.set_word_cost_overrides(HashMap::from([(999_u32, 100_i16)]));
        dict.set_word_cost_overrides(HashMap::new()); // clear

        let after_clear = dict.lookup("すもも");
        assert_eq!(
            base.len(),
            after_clear.len(),
            "lookup length must be identical after clearing overrides"
        );
        for (a, b) in base.iter().zip(after_clear.iter()) {
            assert_eq!(a.wcost, b.wcost, "wcost must be unchanged after clearing overrides");
            assert_eq!(a.word_id, b.word_id);
        }
    }

    /// Verify that a word-cost override is correctly applied to matching entries.
    #[test]
    fn test_word_cost_override_applied() {
        use mecrab_builder::build_synthetic_dictionary;
        let sd = build_synthetic_dictionary();
        let dict = Dictionary::from_bytes(&sd.sys_dic, &sd.matrix, &sd.char_def, &sd.unk_def)
            .expect("build dict");

        // Find a word_id that appears in the lookup results for "すもも"
        let base = dict.lookup("すもも");
        if base.is_empty() {
            // If the synthetic dict doesn't have "すもも", skip gracefully
            return;
        }
        let target = base[0].clone();
        let delta: i16 = -200;
        let expected_wcost = (target.wcost as i64 + delta as i64)
            .clamp(i16::MIN as i64, i16::MAX as i64) as i16;

        dict.set_word_cost_overrides(HashMap::from([(target.word_id, delta)]));
        let after = dict.lookup("すもも");

        let overridden = after.iter().find(|e| e.word_id == target.word_id)
            .expect("overridden word must still appear in results");
        assert_eq!(
            overridden.wcost, expected_wcost,
            "wcost should be adjusted by delta"
        );

        // Non-targeted word_id should be unaffected
        for e in &after {
            if e.word_id != target.word_id {
                let base_entry = base.iter().find(|b| b.word_id == e.word_id);
                if let Some(b) = base_entry {
                    assert_eq!(e.wcost, b.wcost, "unaffected word_id must keep original wcost");
                }
            }
        }
    }

    /// Verify load_word_cost_overrides parses TSV correctly.
    #[test]
    fn test_load_word_cost_overrides_tsv() {
        use mecrab_builder::build_synthetic_dictionary;
        use std::env::temp_dir;
        use std::io::Write;

        let sd = build_synthetic_dictionary();
        let dict = Dictionary::from_bytes(&sd.sys_dic, &sd.matrix, &sd.char_def, &sd.unk_def)
            .expect("build dict");

        let path = temp_dir().join("mecrab_test_override.tsv");
        {
            let mut f = std::fs::File::create(&path).expect("create tsv");
            writeln!(f, "# comment line").unwrap();
            writeln!(f, "42\t-100").unwrap();
            writeln!(f, "99\t200").unwrap();
        }
        dict.load_word_cost_overrides(&path).expect("load_word_cost_overrides");
        assert_eq!(dict.override_count.load(Ordering::Acquire), 2);
        {
            let map = dict.word_cost_overrides.read().unwrap();
            assert_eq!(map.get(&42).copied(), Some(-100_i16));
            assert_eq!(map.get(&99).copied(), Some(200_i16));
        }
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_load_ipadic() {
        let path = Path::new("/var/lib/mecab/dic/ipadic-utf8");
        if !path.exists() {
            eprintln!("IPADIC not found, skipping test");
            return;
        }

        let dict = Dictionary::load(path).expect("Failed to load dictionary");

        eprintln!("Dictionary loaded:");
        eprintln!("  Size: {} entries", dict.size());
        eprintln!("  Charset: {}", dict.charset());

        // Test lookup for 'の'
        let entries = dict.lookup("の");
        eprintln!("\nLookup 'の': {} entries", entries.len());
        for entry in entries.iter().take(3) {
            eprintln!(
                "  length={}, wcost={}, feature={}",
                entry.length, entry.wcost, entry.feature
            );
        }
        assert!(!entries.is_empty(), "Expected to find 'の' in dictionary");

        // Test lookup for 'テスト'
        let entries = dict.lookup("テスト");
        eprintln!("\nLookup 'テスト': {} entries", entries.len());
        for entry in entries.iter().take(3) {
            eprintln!(
                "  length={}, wcost={}, feature={}",
                entry.length, entry.wcost, entry.feature
            );
        }
        assert!(
            !entries.is_empty(),
            "Expected to find 'テスト' in dictionary"
        );

        // Test lookup for '東京'
        let entries = dict.lookup("東京");
        eprintln!("\nLookup '東京': {} entries", entries.len());
        for entry in entries.iter().take(3) {
            eprintln!(
                "  length={}, wcost={}, feature={}",
                entry.length, entry.wcost, entry.feature
            );
        }
        assert!(!entries.is_empty(), "Expected to find '東京' in dictionary");
    }
}
