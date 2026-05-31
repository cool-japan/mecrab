//! Overlay dictionary for live word additions
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides an in-memory dictionary layer that sits on top of
//! the static system dictionary, allowing runtime word additions without
//! restarting or reloading the main dictionary.
//!
//! Architecture:
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           Overlay Dictionary             │ ← API updates (hot path)
//! ├─────────────────────────────────────────┤
//! │           System Dictionary              │ ← IPADIC (mmap, read-only)
//! └─────────────────────────────────────────┘
//! ```
//!
//! Implementation uses yada (double-array trie) for fast prefix lookups
//! when the dictionary is stable, with fallback to HashMap during updates.

use crate::dict::DictionaryEntry;
use std::collections::HashMap;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use yada::DoubleArray;
use yada::builder::DoubleArrayBuilder;

/// Surface to index mapping for trie lookup
type SurfaceIndex = HashMap<String, usize>;

/// An in-memory overlay dictionary for runtime word additions
///
/// This structure allows adding new words at runtime without modifying
/// the underlying system dictionary. Lookups check the overlay first,
/// then fall back to the system dictionary.
///
/// Uses a double-array trie (yada) for efficient prefix lookups.
/// The trie is rebuilt when words are added/removed.
pub struct OverlayDictionary {
    /// Word entries indexed by surface form (primary storage)
    /// Using RwLock for thread-safe concurrent access
    entries: RwLock<HashMap<String, Vec<OverlayEntry>>>,
    /// Double-array trie for fast prefix lookup
    /// Lazily rebuilt after modifications
    trie: RwLock<Option<DoubleArray<Vec<u8>>>>,
    /// Mapping from surface to sorted index for trie value resolution
    surface_index: RwLock<SurfaceIndex>,
    /// Sorted surface list for index resolution
    sorted_surfaces: RwLock<Vec<String>>,
    /// Flag indicating trie needs rebuild
    trie_dirty: RwLock<bool>,
    /// Total number of individual `OverlayEntry` records stored.
    ///
    /// Maintained atomically so that `is_empty()` can perform a lock-free
    /// check with a single `load(Relaxed)` — no RwLock acquisition required
    /// in the common (empty-overlay) fast path.
    entry_count: AtomicUsize,
}

/// A single entry in the overlay dictionary
#[derive(Debug, Clone)]
pub struct OverlayEntry {
    /// Left context ID for connection matrix
    pub left_id: u16,
    /// Right context ID for connection matrix
    pub right_id: u16,
    /// Word cost (lower = more preferred)
    pub wcost: i16,
    /// Feature string (POS info, reading, etc.)
    pub feature: String,
}

impl OverlayEntry {
    /// Create a new overlay entry with automatic context IDs
    ///
    /// Uses default context IDs (1285 = 名詞,一般 in IPADIC)
    pub fn new(feature: &str, wcost: i16) -> Self {
        Self {
            left_id: 1285, // 名詞,一般 (common noun)
            right_id: 1285,
            wcost,
            feature: feature.to_string(),
        }
    }

    /// Create an entry with explicit context IDs
    pub fn with_context(left_id: u16, right_id: u16, wcost: i16, feature: &str) -> Self {
        Self {
            left_id,
            right_id,
            wcost,
            feature: feature.to_string(),
        }
    }
}

impl Default for OverlayDictionary {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlayDictionary {
    /// Create a new empty overlay dictionary
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            trie: RwLock::new(None),
            surface_index: RwLock::new(HashMap::new()),
            sorted_surfaces: RwLock::new(Vec::new()),
            trie_dirty: RwLock::new(false),
            entry_count: AtomicUsize::new(0),
        }
    }

    /// Add a word to the overlay dictionary
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form (the actual text)
    /// * `entry` - The dictionary entry data
    ///
    /// # Example
    ///
    /// ```ignore
    /// let overlay = OverlayDictionary::new();
    /// overlay.add_word("ChatGPT", OverlayEntry::new(
    ///     "名詞,固有名詞,一般,*,*,*,ChatGPT,チャットジーピーティー,チャットジーピーティー",
    ///     5000,
    /// ));
    /// ```
    pub fn add_word(&self, surface: &str, entry: OverlayEntry) {
        {
            let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
            entries.entry(surface.to_string()).or_default().push(entry);
        }
        // Increment the lock-free counter used by `is_empty()` / `len()`.
        self.entry_count.fetch_add(1, Ordering::Release);
        // Mark trie as dirty
        *self.trie_dirty.write().unwrap_or_else(|e| e.into_inner()) = true;
    }

    /// Add a word with simple parameters
    ///
    /// This is a convenience method that creates a common noun entry.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface form
    /// * `reading` - The katakana reading
    /// * `pronunciation` - The pronunciation (often same as reading)
    /// * `wcost` - Word cost (lower = more preferred, typical: 5000-8000)
    pub fn add_simple(&self, surface: &str, reading: &str, pronunciation: &str, wcost: i16) {
        let feature = format!(
            "名詞,固有名詞,一般,*,*,*,{},{},{}",
            surface, reading, pronunciation
        );
        self.add_word(surface, OverlayEntry::new(&feature, wcost));
    }

    /// Remove a word from the overlay dictionary
    ///
    /// Returns true if the word was found and removed.
    pub fn remove_word(&self, surface: &str) -> bool {
        let removed_count = {
            let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
            entries.remove(surface).map(|v| v.len()).unwrap_or(0)
        };
        if removed_count > 0 {
            // Keep the atomic counter consistent with the actual entry set.
            self.entry_count.fetch_sub(removed_count, Ordering::Release);
            *self.trie_dirty.write().unwrap_or_else(|e| e.into_inner()) = true;
            true
        } else {
            false
        }
    }

    /// Rebuild the trie from current entries
    fn rebuild_trie(&self) {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());

        if entries.is_empty() {
            *self.trie.write().unwrap_or_else(|e| e.into_inner()) = None;
            *self
                .surface_index
                .write()
                .unwrap_or_else(|e| e.into_inner()) = HashMap::new();
            *self
                .sorted_surfaces
                .write()
                .unwrap_or_else(|e| e.into_inner()) = Vec::new();
            *self.trie_dirty.write().unwrap_or_else(|e| e.into_inner()) = false;
            return;
        }

        // Build sorted key-value pairs for yada
        // Keys must be sorted for yada to work correctly
        let mut surfaces: Vec<String> = entries.keys().cloned().collect();
        surfaces.sort();

        // Build keyset: (key_bytes, index)
        let keyset: Vec<(&[u8], u32)> = surfaces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_bytes(), i as u32))
            .collect();

        // Build surface index for reverse lookup
        let new_index: SurfaceIndex = surfaces
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        // Build the trie
        if let Some(da_bytes) = DoubleArrayBuilder::build(&keyset) {
            *self.trie.write().unwrap_or_else(|e| e.into_inner()) =
                Some(DoubleArray::new(da_bytes));
            *self
                .surface_index
                .write()
                .unwrap_or_else(|e| e.into_inner()) = new_index;
            *self
                .sorted_surfaces
                .write()
                .unwrap_or_else(|e| e.into_inner()) = surfaces;
        }
        *self.trie_dirty.write().unwrap_or_else(|e| e.into_inner()) = false;
    }

    /// Look up a word in the overlay dictionary
    ///
    /// Returns all entries that match the given key as a prefix.
    /// Uses double-array trie for O(m) prefix matching where m is key length.
    pub fn lookup(&self, key: &str) -> Vec<DictionaryEntry> {
        // Check if trie needs rebuild
        {
            let dirty = *self.trie_dirty.read().unwrap_or_else(|e| e.into_inner());
            if dirty {
                drop(self.trie_dirty.read());
                self.rebuild_trie();
            }
        }

        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        if entries.is_empty() {
            return Vec::new();
        }

        let trie_guard = self.trie.read().unwrap_or_else(|e| e.into_inner());
        let sorted_surfaces = self
            .sorted_surfaces
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let mut results = Vec::new();

        if let Some(ref trie) = *trie_guard {
            // Use trie for common prefix search
            let key_bytes = key.as_bytes();
            for (value, length) in trie.common_prefix_search(key_bytes) {
                // Get surface from sorted list using trie value as index
                if let Some(surface) = sorted_surfaces.get(value as usize) {
                    if let Some(entry_list) = entries.get(surface) {
                        for entry in entry_list {
                            results.push(DictionaryEntry {
                                length,
                                word_id: u32::MAX, // Overlay entries don't have dictionary word_id
                                left_id: entry.left_id,
                                right_id: entry.right_id,
                                pos_id: entry.left_id,
                                wcost: entry.wcost,
                                feature: entry.feature.clone(),
                            });
                        }
                    }
                }
            }
        } else {
            // Fallback to HashMap scan (rare case when trie build fails)
            for (surface, surface_entries) in entries.iter() {
                if key.starts_with(surface) {
                    for entry in surface_entries {
                        results.push(DictionaryEntry {
                            length: surface.len(),
                            word_id: u32::MAX, // Overlay entries don't have dictionary word_id
                            left_id: entry.left_id,
                            right_id: entry.right_id,
                            pos_id: entry.left_id,
                            wcost: entry.wcost,
                            feature: entry.feature.clone(),
                        });
                    }
                }
            }
        }

        results
    }

    /// Get the number of entries in the overlay
    ///
    /// Uses the `AtomicUsize` counter for O(1) performance — no RwLock acquired.
    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Acquire)
    }

    /// Returns `true` when no words have been added to the overlay.
    ///
    /// This is a lock-free O(1) check using an `AtomicUsize` counter —
    /// no `RwLock` is acquired.  It is safe to call from the hot `lookup`
    /// fast-path with negligible overhead.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entry_count.load(Ordering::Acquire) == 0
    }

    /// Clear all entries from the overlay
    pub fn clear(&self) {
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        entries.clear();
        // Reset the lock-free counter together with the entries map.
        self.entry_count.store(0, Ordering::Release);
        *self.trie.write().unwrap_or_else(|e| e.into_inner()) = None;
        *self
            .surface_index
            .write()
            .unwrap_or_else(|e| e.into_inner()) = HashMap::new();
        *self
            .sorted_surfaces
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Vec::new();
        *self.trie_dirty.write().unwrap_or_else(|e| e.into_inner()) = false;
    }

    /// Get all surface forms in the overlay
    pub fn surfaces(&self) -> Vec<String> {
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        entries.keys().cloned().collect()
    }

    /// Export all overlay entries for preservation across a dictionary hot-swap.
    pub fn snapshot(&self) -> Vec<(String, Vec<OverlayEntry>)> {
        self.entries
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Merge a snapshot of entries back in (used after hot-swap to restore overlay words).
    pub fn restore_from(&self, snapshot: Vec<(String, Vec<OverlayEntry>)>) {
        if snapshot.is_empty() {
            return;
        }
        let mut added: usize = 0;
        {
            let mut guard = self.entries.write().unwrap_or_else(|e| e.into_inner());
            for (surface, entries) in snapshot {
                added += entries.len();
                guard.entry(surface).or_default().extend(entries);
            }
        }
        if added > 0 {
            self.entry_count.fetch_add(added, Ordering::Release);
        }
        // Mark trie as dirty so it is rebuilt on next lookup.
        if let Ok(mut dirty) = self.trie_dirty.write() {
            *dirty = true;
        }
    }
}

impl std::fmt::Debug for OverlayDictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OverlayDictionary")
            .field("count", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_lookup() {
        let overlay = OverlayDictionary::new();

        overlay.add_simple(
            "ChatGPT",
            "チャットジーピーティー",
            "チャットジーピーティー",
            5000,
        );

        let results = overlay.lookup("ChatGPT");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].length, "ChatGPT".len());
        assert_eq!(results[0].wcost, 5000);
        assert!(results[0].feature.contains("ChatGPT"));
    }

    #[test]
    fn test_remove_word() {
        let overlay = OverlayDictionary::new();

        overlay.add_simple("テスト", "テスト", "テスト", 5000);
        assert_eq!(overlay.len(), 1);

        assert!(overlay.remove_word("テスト"));
        assert_eq!(overlay.len(), 0);

        assert!(!overlay.remove_word("テスト"));
    }

    #[test]
    fn test_prefix_lookup() {
        let overlay = OverlayDictionary::new();

        overlay.add_simple("東京", "トウキョウ", "トーキョー", 5000);

        // Should match "東京都" because "東京" is a prefix
        let results = overlay.lookup("東京都");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].length, "東京".len());
    }

    #[test]
    fn test_multiple_entries() {
        let overlay = OverlayDictionary::new();

        overlay.add_simple("日本", "ニホン", "ニホン", 5000);
        overlay.add_simple("日本語", "ニホンゴ", "ニホンゴ", 5000);

        // Should find both when looking up "日本語学校"
        let results = overlay.lookup("日本語学校");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let overlay = Arc::new(OverlayDictionary::new());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let overlay = Arc::clone(&overlay);
                thread::spawn(move || {
                    let surface = format!("word{}", i);
                    overlay.add_simple(&surface, "ワード", "ワード", 5000);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(overlay.len(), 10);
    }

    #[test]
    fn test_trie_rebuild() {
        let overlay = OverlayDictionary::new();

        // Add multiple words to trigger trie build
        overlay.add_simple("Apple", "アップル", "アップル", 5000);
        overlay.add_simple("Banana", "バナナ", "バナナ", 5000);
        overlay.add_simple("Cherry", "チェリー", "チェリー", 5000);

        // First lookup triggers trie build
        let results = overlay.lookup("Apple");
        assert_eq!(results.len(), 1);

        // Subsequent lookups use the trie
        let results = overlay.lookup("Banana");
        assert_eq!(results.len(), 1);
    }

    /// Verify that `is_empty()` tracks the full lifecycle using the atomic counter:
    /// starts true, becomes false after `add_word`, returns true after removing all words.
    #[test]
    fn test_is_empty_lifecycle() {
        let overlay = OverlayDictionary::new();

        // Fresh overlay must be empty
        assert!(overlay.is_empty(), "new overlay should be empty");
        assert_eq!(overlay.len(), 0);

        overlay.add_simple("テスト", "テスト", "テスト", 5000);
        assert!(!overlay.is_empty(), "overlay should not be empty after add");
        assert_eq!(overlay.len(), 1);

        // Add a second entry under the same surface key
        overlay.add_word(
            "テスト",
            OverlayEntry::with_context(100, 100, 6000, "名詞,サ変接続,*,*,*,*,テスト,テスト,テスト"),
        );
        assert_eq!(overlay.len(), 2, "two entries for same surface");

        // Remove the surface — both entries should disappear atomically
        assert!(overlay.remove_word("テスト"));
        assert!(overlay.is_empty(), "overlay should be empty after remove");
        assert_eq!(overlay.len(), 0);
    }

    /// Verify that `clear()` resets the atomic counter.
    #[test]
    fn test_clear_resets_counter() {
        let overlay = OverlayDictionary::new();

        overlay.add_simple("Apple", "アップル", "アップル", 5000);
        overlay.add_simple("Banana", "バナナ", "バナナ", 5000);
        assert_eq!(overlay.len(), 2);
        assert!(!overlay.is_empty());

        overlay.clear();
        assert_eq!(overlay.len(), 0);
        assert!(overlay.is_empty());
    }

    /// Verify that `restore_from` keeps the atomic counter consistent.
    #[test]
    fn test_restore_from_updates_counter() {
        let src = OverlayDictionary::new();
        src.add_simple("東京", "トウキョウ", "トーキョー", 5000);
        src.add_simple("大阪", "オオサカ", "オオサカ", 5000);

        let snap = src.snapshot();
        assert_eq!(snap.len(), 2);

        let dst = OverlayDictionary::new();
        assert!(dst.is_empty());
        dst.restore_from(snap);
        assert_eq!(dst.len(), 2);
        assert!(!dst.is_empty());
    }
}
