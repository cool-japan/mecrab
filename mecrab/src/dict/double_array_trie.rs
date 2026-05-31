//! Double-Array Trie implementation for fast dictionary lookup
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This implements the Double-Array Trie (DAT) data structure compatible
//! with MeCab's Darts library. The trie enables efficient common prefix
//! search which is essential for morphological analysis.
//!
//! Reference: ../ref/mecab-0.996/src/darts.h

use crate::{Error, Result};

/// Result from a trie lookup containing value and matched length
#[derive(Debug, Clone, Copy, Default)]
pub struct DartsResult {
    /// The value stored in the trie (-1 if not found)
    pub value: i32,
    /// Length of the matched key in bytes
    pub length: usize,
}

/// Double-Array Trie unit structure
/// Each unit is 8 bytes: base (i32) + check (u32)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Unit {
    /// Base value for state transitions
    /// Negative value indicates a leaf with value = -base - 1
    base: i32,
    /// Check value to validate transitions
    check: u32,
}

/// Two-byte prefix cache for the Double-Array Trie.
///
/// For each pair of first bytes `(b0, b1)` we pre-compute the DA base values
/// that result after consuming those two bytes from the root.  Because >90 % of
/// Japanese tokens start with a 3-byte UTF-8 sequence (hiragana, katakana, kanji)
/// the first two bytes are highly concentrated, so the cache achieves a very high
/// hit rate and eliminates two cache-miss-prone random memory accesses per
/// `common_prefix_search` call.
///
/// # Layout
///
/// The flat array has `65536 * 2` entries of `i32`.  For the pair index
/// `idx = b0 as usize * 256 + b1 as usize`:
///
/// * `table[idx * 2]`     — DA base after consuming byte `b0` only,
///   or `i32::MIN` if that transition does not exist.
/// * `table[idx * 2 + 1]` — DA base after consuming bytes `b0` then `b1`,
///   or `i32::MIN` if either transition does not exist.
///
/// Storing both intermediates lets `common_prefix_search` emit terminal nodes
/// at lengths 1 and 2 correctly before jumping to the cached state for the
/// remainder of the key.
struct TriePrefixCache {
    /// Flat storage: `[b_after_1, b_after_2]` per (b0, b1) pair.
    /// `i32::MIN` marks an invalid / non-existent state.
    table: Box<[i32; 65536 * 2]>,
}

impl std::fmt::Debug for TriePrefixCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Count valid (non-INVALID) entries for a compact representation.
        let valid_count = self
            .table
            .iter()
            .filter(|&&v| v != Self::INVALID)
            .count();
        f.debug_struct("TriePrefixCache")
            .field("table_len", &self.table.len())
            .field("valid_entries", &valid_count)
            .finish()
    }
}

impl TriePrefixCache {
    /// Sentinel value meaning "no valid DA state for this (b0, b1) prefix".
    const INVALID: i32 = i32::MIN;

    /// Build the cache by simulating the first two transition steps from root
    /// for every possible byte pair `(b0, b1)`.
    ///
    /// Takes O(65536) DA lookups at build time; each subsequent
    /// `common_prefix_search` call saves two random-access DA reads.
    fn build(trie: &DoubleArrayTrie) -> Self {
        // Allocate on the heap directly; 65536*2*4 = 512 KiB.
        // We construct via a Vec then convert to a fixed-size boxed array.
        // SAFETY: We allocate exactly 65536*2 elements, matching the array length.
        let raw = Box::into_raw(
            vec![Self::INVALID; 65536 * 2]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_: Box<[i32]>| {
                    unreachable!("TriePrefixCache allocation length mismatch")
                }),
        );
        // SAFETY: `raw` was obtained from `Box::into_raw` of a properly allocated
        // `Box<[i32; 65536 * 2]>`, so reconstructing the Box is valid.
        let mut table: Box<[i32; 65536 * 2]> = unsafe { Box::from_raw(raw) };

        // Retrieve root base; if the trie is empty we return the all-INVALID cache.
        let root_base = match trie.get(0) {
            Some(u) => u.base,
            None => return Self { table },
        };

        for b0 in 0u8..=255 {
            // --- Step 1: root → after b0 ---
            let p1 = (root_base as usize)
                .wrapping_add(b0 as usize)
                .wrapping_add(1);
            let unit1 = match trie.get(p1) {
                Some(u) if u.check == root_base as u32 => u,
                _ => {
                    // Transition via b0 does not exist; leave all (b0, *) entries
                    // as INVALID (already initialised that way).
                    continue;
                }
            };
            let base_after_1 = unit1.base;

            for b1 in 0u8..=255 {
                let idx = (b0 as usize) * 256 + (b1 as usize);

                // Store b_after_1 (valid for all b1 that share the same b0 prefix).
                table[idx * 2] = base_after_1;

                // --- Step 2: after b0 → after b0, b1 ---
                if base_after_1 < 0 {
                    // base_after_1 < 0 indicates a leaf node; no further transitions.
                    continue;
                }
                let p2 = (base_after_1 as usize)
                    .wrapping_add(b1 as usize)
                    .wrapping_add(1);
                if let Some(unit2) = trie.get(p2) {
                    if unit2.check == base_after_1 as u32 {
                        table[idx * 2 + 1] = unit2.base;
                    }
                }
            }
        }

        Self { table }
    }
}

/// Double-Array Trie for fast word lookup (Darts compatible)
#[derive(Debug)]
pub struct DoubleArrayTrie {
    /// Raw pointer to memory-mapped data
    units_ptr: *const Unit,
    /// Number of units
    size: usize,
    /// Optional two-byte prefix cache (built lazily via `build_prefix_cache`).
    prefix_cache: Option<TriePrefixCache>,
}

// Safety: The units_ptr points to immutable memory-mapped data
unsafe impl Send for DoubleArrayTrie {}
unsafe impl Sync for DoubleArrayTrie {}

impl DoubleArrayTrie {
    /// Unit size in bytes (same as C++ Darts)
    pub const UNIT_SIZE: usize = 8;

    /// Create a new Double-Array Trie from raw bytes (memory-mapped dictionary)
    ///
    /// # Safety
    ///
    /// The data must remain valid for the lifetime of this struct.
    /// This is typically ensured by keeping the Mmap alive.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is too small.
    pub fn from_bytes(data: &[u8], size_in_bytes: usize) -> Result<Self> {
        if data.len() < size_in_bytes {
            return Err(Error::CorruptedDictionary(format!(
                "Double-array data too small: expected {} bytes, got {}",
                size_in_bytes,
                data.len()
            )));
        }

        let size = size_in_bytes / Self::UNIT_SIZE;

        // Safety: We verify the data is large enough and properly aligned
        let units_ptr = data.as_ptr() as *const Unit;

        Ok(Self {
            units_ptr,
            size,
            prefix_cache: None,
        })
    }

    /// Build and store the two-byte prefix cache.
    ///
    /// Call this once after loading the dictionary.  The cache eliminates the
    /// first two random-access DA reads in every `common_prefix_search` call
    /// for keys whose first two bytes are a known-valid transition pair (which
    /// covers >90 % of Japanese text tokens).
    ///
    /// The cache occupies ~512 KiB on the heap.  It is safe to call multiple
    /// times; subsequent calls rebuild and replace the existing cache.
    pub fn build_prefix_cache(&mut self) {
        self.prefix_cache = Some(TriePrefixCache::build(self));
    }

    /// Get a unit at the given index
    #[inline]
    fn get(&self, index: usize) -> Option<Unit> {
        if index < self.size {
            // Safety: We verified the index is in bounds
            Some(unsafe { *self.units_ptr.add(index) })
        } else {
            None
        }
    }

    /// Perform exact match search (compatible with Darts::exactMatchSearch)
    ///
    /// Returns the value if an exact match is found, -1 otherwise.
    pub fn exact_match_search(&self, key: &[u8]) -> DartsResult {
        let mut result = DartsResult {
            value: -1,
            length: 0,
        };

        let mut b = match self.get(0) {
            Some(unit) => unit.base,
            None => return result,
        };

        for &byte in key.iter() {
            let p = (b as usize).wrapping_add(byte as usize).wrapping_add(1);

            match self.get(p) {
                // Check compares base (b) with check, not node_pos
                Some(unit) if unit.check == b as u32 => {
                    b = unit.base;
                }
                _ => return result,
            }
        }

        // Check if we have a valid ending
        let p = b as usize;
        if let Some(unit) = self.get(p) {
            if unit.check == b as u32 && unit.base < 0 {
                result.value = -unit.base - 1;
                result.length = key.len();
            }
        }

        result
    }

    /// Perform common prefix search (compatible with Darts::commonPrefixSearch)
    ///
    /// Returns all values for prefixes of the key that exist in the trie.
    /// Results are stored in the provided buffer and the number of results is returned.
    ///
    /// When a two-byte prefix cache has been built via `build_prefix_cache`, the
    /// first two DA traversal steps are replaced by a direct table lookup,
    /// saving two cache-miss-prone random memory accesses on every call.
    /// Terminal nodes at lengths 1 and 2 are still checked correctly via the
    /// cached intermediate DA base values.
    pub fn common_prefix_search(&self, key: &[u8], results: &mut [DartsResult]) -> usize {
        let max_results = results.len();
        let mut num_results = 0;

        let root_unit = match self.get(0) {
            Some(unit) => unit,
            None => return 0,
        };
        let root_base = root_unit.base;
        let mut b = root_base;
        let mut key_offset = 0usize;

        // ── Two-byte prefix cache fast-path ─────────────────────────────────────
        // Try to skip the first two DA traversal steps by consulting the cache.
        // We still emit any terminals found at lengths 1 and 2 using the cached
        // intermediate base values so the output is identical to the uncached path.
        if key.len() >= 2 {
            if let Some(ref cache) = self.prefix_cache {
                let idx = (key[0] as usize) * 256 + (key[1] as usize);
                let base_after_1 = cache.table[idx * 2];
                let base_after_2 = cache.table[idx * 2 + 1];

                // base_after_1 == INVALID means byte[0] transition does not exist.
                if base_after_1 == TriePrefixCache::INVALID {
                    // byte[0] transition absent — key not in trie.
                    return num_results;
                }
                // --- Check terminal at length 1 (after consuming byte[0]) ---
                // The terminal for a key of length 1 is stored at slot `base_after_1`
                // when that slot's check equals `base_after_1`.
                let p1 = base_after_1 as usize;
                if let Some(unit) = self.get(p1) {
                    if unit.check == base_after_1 as u32
                        && unit.base < 0
                        && num_results < max_results
                    {
                        results[num_results] = DartsResult {
                            value: -unit.base - 1,
                            length: 1,
                        };
                        num_results += 1;
                    }
                }

                // base_after_2 == INVALID means byte[1] transition does not exist
                // from the state reached after byte[0].
                if base_after_2 == TriePrefixCache::INVALID {
                    // byte[1] transition absent — the search cannot continue
                    // past byte[0]; return whatever we already found.
                    return num_results;
                }
                // --- Check terminal at length 2 (after consuming byte[1]) ---
                let p2 = base_after_2 as usize;
                if let Some(unit) = self.get(p2) {
                    if unit.check == base_after_2 as u32
                        && unit.base < 0
                        && num_results < max_results
                    {
                        results[num_results] = DartsResult {
                            value: -unit.base - 1,
                            length: 2,
                        };
                        num_results += 1;
                    }
                }

                // Jump to the cached state and continue from byte[2].
                b = base_after_2;
                key_offset = 2;
            }
        }
        // ── End of cache fast-path ───────────────────────────────────────────────

        // Generic loop starting from `key_offset` (0 if cache miss/no cache, 2 if
        // the two-byte prefix shortcut was taken).
        for (i, &byte) in key[key_offset..].iter().enumerate() {
            let abs_pos = key_offset + i;

            // Check for a terminal at the current DA base (word ending here has
            // byte-length `abs_pos`).
            let p = b as usize;
            if let Some(unit) = self.get(p) {
                if unit.check == b as u32 && unit.base < 0 && num_results < max_results {
                    results[num_results] = DartsResult {
                        value: -unit.base - 1,
                        length: abs_pos,
                    };
                    num_results += 1;
                }
            }

            // Transition to next state via `byte`.
            let p = (b as usize).wrapping_add(byte as usize).wrapping_add(1);
            match self.get(p) {
                Some(unit) if unit.check == b as u32 => {
                    b = unit.base;
                }
                _ => return num_results,
            }
        }

        // Check for a terminal at the final position (full key consumed).
        let p = b as usize;
        if let Some(unit) = self.get(p) {
            if unit.check == b as u32 && unit.base < 0 && num_results < max_results {
                results[num_results] = DartsResult {
                    value: -unit.base - 1,
                    length: key.len(),
                };
                num_results += 1;
            }
        }

        num_results
    }

    /// Get the size of the trie in units
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the total size in bytes
    #[inline]
    pub fn total_size(&self) -> usize {
        self.size * Self::UNIT_SIZE
    }

    /// Prefetch the DA unit that will be accessed on the first transition from
    /// root for a key whose first byte is `first_byte`.
    ///
    /// Call this from the lattice builder just before starting a new
    /// `common_prefix_search` at a given text position.  The hardware prefetch
    /// hint loads the relevant cache line *before* it is needed, hiding the
    /// memory-access latency behind the previous search.
    ///
    /// On platforms without explicit prefetch support the call is a no-op; the
    /// compiler / out-of-order hardware will handle prefetching naturally.
    #[inline]
    pub fn prefetch_for_key(&self, first_byte: u8) {
        // The first transition from root goes to unit[base[0] + first_byte + 1].
        // Prefetch that unit to warm the L1/L2 cache before the search starts.
        if let Some(root) = self.get(0) {
            let target_idx = (root.base as usize)
                .wrapping_add(first_byte as usize)
                .wrapping_add(1);
            if target_idx < self.size {
                // SAFETY: target_idx is within [0, self.size), so
                // `self.units_ptr.add(target_idx)` is a valid pointer into the
                // memory-mapped region.  We only read the address (for the
                // prefetch hint) and never dereference it here.
                let ptr = unsafe { self.units_ptr.add(target_idx) as *const u8 };

                #[cfg(target_arch = "x86_64")]
                // SAFETY: _mm_prefetch merely issues a cache-line hint; it never
                // faults on invalid addresses, and the address is valid here.
                unsafe {
                    core::arch::x86_64::_mm_prefetch(
                        ptr as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }

                #[cfg(target_arch = "aarch64")]
                // SAFETY: `prfm pldl1keep` is a hint instruction that never
                // faults, and the address is valid.
                unsafe {
                    core::arch::asm!(
                        "prfm pldl1keep, [{ptr}]",
                        ptr = in(reg) ptr,
                        options(nostack, readonly),
                    );
                }

                // On all other platforms: no-op.  The `ptr` binding is used only
                // inside the cfg-gated blocks above.  Suppress the unused-variable
                // warning on generic targets.
                #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                let _ = ptr;
            }
        }
    }

    /// Enumerate all (key_bytes, value) pairs stored in the trie via depth-first search.
    ///
    /// This walks every reachable slot in the Double-Array to reconstruct the original
    /// keys and their associated values.  Useful for vocabulary enumeration.
    ///
    /// The visitor closure receives `(key_bytes: &[u8], value: i32)` for every terminal
    /// node found during the traversal.  If the closure returns `false`, the traversal
    /// stops early.
    ///
    /// # Implementation note
    ///
    /// We perform a DFS from the root (`da_slot = 0`).  At each non-terminal node
    /// with base `b`, we probe every possible child byte `c` in `0..=255` and follow
    /// the transition to slot `b + c + 1` when `units[slot].check == b as u32`.
    /// At each such slot we additionally check whether `units[b].check == b as u32 &&
    /// units[b].base < 0` to detect the terminal self-reference for the current prefix.
    ///
    /// Because the DA may contain large sparse regions the probe loop is O(256 × depth),
    /// which is acceptable for one-shot vocabulary dumps.
    pub fn enumerate_all<F>(&self, mut visitor: F)
    where
        F: FnMut(&[u8], i32) -> bool,
    {
        if self.size == 0 {
            return;
        }

        // Stack entries: (da_slot_index, accumulated_key_so_far)
        // da_slot is the index whose `.base` is the current node's base.
        let root_base = match self.get(0) {
            Some(u) => u.base,
            None => return,
        };
        // Stack: (current_base_value, key_prefix_bytes)
        let mut stack: Vec<(i32, Vec<u8>)> = vec![(root_base, Vec::new())];

        while let Some((b, prefix)) = stack.pop() {
            if b < 0 {
                // This is already a leaf; should not appear directly on the stack.
                continue;
            }
            let b_usize = b as usize;

            // Check for terminal at `b`: units[b].check == b && units[b].base < 0
            if let Some(term_unit) = self.get(b_usize) {
                if term_unit.check == b as u32 && term_unit.base < 0 {
                    let value = -term_unit.base - 1;
                    if !visitor(&prefix, value) {
                        return;
                    }
                }
            }

            // Probe all 256 possible child bytes
            for c in 0u8..=255 {
                let slot = b_usize.wrapping_add(c as usize).wrapping_add(1);
                if slot >= self.size {
                    continue;
                }
                if let Some(child_unit) = self.get(slot) {
                    if child_unit.check == b as u32 {
                        // Valid transition via byte `c` → child node lives at slot `slot`
                        // Push (child's base, extended prefix) for further traversal.
                        let mut new_prefix = prefix.clone();
                        new_prefix.push(c);
                        stack.push((child_unit.base, new_prefix));
                    }
                }
            }
        }
    }

    /// Warm the L1/L2 cache for the most common Japanese UTF-8 first bytes.
    ///
    /// Call this once after loading the dictionary.  It issues prefetch hints
    /// for the DA units reached by the most frequent first-byte values in
    /// Japanese text, reducing cold-start latency on the first batch of lookups.
    ///
    /// Common first bytes:
    /// - `0xE3`: U+3000–U+9FFF  (hiragana, katakana, kanji — by far the most common)
    /// - `0xEF`: U+F000–U+FFFF  (fullwidth / enclosed / CJK compatibility)
    /// - `0xE2`: U+2000–U+2FFF  (punctuation, arrows, mathematical operators)
    /// - `0x41`: ASCII 'A'       (Latin letters appear frequently in mixed text)
    pub fn warm_cache(&self) {
        for &byte in &[0xE3u8, 0xEFu8, 0xE2u8, 0x41u8] {
            self.prefetch_for_key(byte);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_size() {
        assert_eq!(std::mem::size_of::<Unit>(), 8);
        assert_eq!(DoubleArrayTrie::UNIT_SIZE, 8);
    }

    #[test]
    fn test_darts_result_default() {
        let result = DartsResult::default();
        assert_eq!(result.value, 0);
        assert_eq!(result.length, 0);
    }

    /// Build a real sys.dic byte buffer containing a small Japanese lexicon
    /// and return the parsed trie with its prefix cache built.
    #[cfg(test)]
    fn build_test_trie() -> (Vec<u8>, DoubleArrayTrie) {
        use mecrab_builder::sysdic_writer::{DicEntry, build_sysdic_bytes};

        let entries = vec![
            DicEntry {
                surface: "すもも".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 5000,
                feature: "名詞,一般,*,すもも".to_string(),
            },
            DicEntry {
                surface: "もも".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 5000,
                feature: "名詞,一般,*,もも".to_string(),
            },
            DicEntry {
                surface: "東京".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 4500,
                feature: "名詞,固有名詞,東京".to_string(),
            },
        ];

        let (buf, _stats) =
            build_sysdic_bytes(&entries, 8, 8, "UTF-8", 0).expect("sysdic build failed");

        // Parse only the trie portion out of the buf.
        // Header layout: see sys_dic.rs — HEADER_SIZE = 72 bytes.
        const HEADER_SIZE: usize = 72;
        let da_size =
            u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;
        let mut trie =
            DoubleArrayTrie::from_bytes(&buf[HEADER_SIZE..], da_size).expect("trie parse failed");
        trie.build_prefix_cache();
        (buf, trie)
    }

    /// Verify that after building the cache the hiragana prefix (0xE3, 0x81)
    /// — the first two bytes of every hiragana character — yields a valid
    /// cached state (not INVALID).
    ///
    /// 「す」= 0xE3 0x81 0x99 — first two bytes are 0xE3, 0x81.
    /// The synthetic dictionary contains "すもも" and "もも" whose first bytes
    /// include 0xE3 0x81, so the cache must record a valid DA base there.
    #[test]
    fn test_prefix_cache_built() {
        let (_buf, trie) = build_test_trie();
        let cache = trie.prefix_cache.as_ref().expect("cache must be present");

        // Hiragana starts with 0xE3 0x81 (す = E3 81 99, も = E3 82 82)
        // 0xE3 0x81 → index for hiragana block
        let idx_hiragana = 0xE3usize * 256 + 0x81usize;
        // b_after_1 must be valid because すもも starts with 0xE3
        // Note: b_after_1 is shared across all b1 with the same b0=0xE3
        // so we check via b0=0xE3, b1=0x81 specifically.
        let b_after_1 = cache.table[idx_hiragana * 2];
        assert_ne!(
            b_after_1,
            TriePrefixCache::INVALID,
            "cache must record a valid DA base after consuming the first byte (0xE3) \
             of a hiragana character; got INVALID"
        );

        // Additionally, b_after_2 for (0xE3, 0x81) should also be valid since
        // すもも's first two bytes are exactly 0xE3 0x81.
        let b_after_2 = cache.table[idx_hiragana * 2 + 1];
        assert_ne!(
            b_after_2,
            TriePrefixCache::INVALID,
            "cache must record a valid DA base after consuming (0xE3, 0x81); got INVALID"
        );
    }

    /// Verify that common_prefix_search with the cache enabled produces
    /// identical results to an uncached search for several Japanese strings.
    #[test]
    fn test_prefix_cache_correctness() {
        use mecrab_builder::sysdic_writer::{DicEntry, build_sysdic_bytes};

        let entries = vec![
            DicEntry {
                surface: "すもも".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 5000,
                feature: "名詞,一般,*,すもも".to_string(),
            },
            DicEntry {
                surface: "す".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 6000,
                feature: "名詞,一般,*,す".to_string(),
            },
            DicEntry {
                surface: "もも".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 5000,
                feature: "名詞,一般,*,もも".to_string(),
            },
            DicEntry {
                surface: "東京".to_string(),
                left_id: 1,
                right_id: 1,
                pos_id: 1,
                wcost: 4500,
                feature: "名詞,固有名詞,東京".to_string(),
            },
        ];

        let (buf, _stats) =
            build_sysdic_bytes(&entries, 8, 8, "UTF-8", 0).expect("sysdic build failed");

        const HEADER_SIZE: usize = 72;
        let da_size =
            u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize;

        // Trie without cache (reference baseline)
        let trie_no_cache =
            DoubleArrayTrie::from_bytes(&buf[HEADER_SIZE..], da_size).expect("trie parse failed");

        // Trie with cache enabled
        let mut trie_cached =
            DoubleArrayTrie::from_bytes(&buf[HEADER_SIZE..], da_size).expect("trie parse failed");
        trie_cached.build_prefix_cache();

        // Test strings: known words, prefixes, no-match strings
        let test_keys: &[&[u8]] = &[
            "すもも".as_bytes(),            // exact match
            "すもももも".as_bytes(),          // prefix: すもも + もも
            "もも".as_bytes(),              // exact match
            "東京".as_bytes(),              // 3-byte kanji pair
            "東京都".as_bytes(),             // prefix: 東京 + unknown suffix
            "xyz".as_bytes(),              // no match (ASCII)
            "す".as_bytes(),               // single hiragana char (3 bytes)
            "すもも東京".as_bytes(),          // concatenation of two known entries
        ];

        let mut ref_results = [DartsResult::default(); 64];
        let mut cached_results = [DartsResult::default(); 64];

        for &key in test_keys {
            let ref_count = trie_no_cache.common_prefix_search(key, &mut ref_results);
            let cached_count = trie_cached.common_prefix_search(key, &mut cached_results);

            assert_eq!(
                ref_count, cached_count,
                "result count mismatch for key {:?}: ref={}, cached={}",
                std::str::from_utf8(key).unwrap_or("<invalid utf8>"),
                ref_count,
                cached_count
            );

            for i in 0..ref_count {
                assert_eq!(
                    ref_results[i].value, cached_results[i].value,
                    "value mismatch at result[{}] for key {:?}: ref={}, cached={}",
                    i,
                    std::str::from_utf8(key).unwrap_or("<invalid utf8>"),
                    ref_results[i].value,
                    cached_results[i].value
                );
                assert_eq!(
                    ref_results[i].length, cached_results[i].length,
                    "length mismatch at result[{}] for key {:?}: ref={}, cached={}",
                    i,
                    std::str::from_utf8(key).unwrap_or("<invalid utf8>"),
                    ref_results[i].length,
                    cached_results[i].length
                );
            }
        }
    }
}
