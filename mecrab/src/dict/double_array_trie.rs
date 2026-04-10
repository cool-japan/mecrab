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

/// Double-Array Trie for fast word lookup (Darts compatible)
#[derive(Debug)]
pub struct DoubleArrayTrie {
    /// Raw pointer to memory-mapped data
    units_ptr: *const Unit,
    /// Number of units
    size: usize,
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

        Ok(Self { units_ptr, size })
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
    pub fn common_prefix_search(&self, key: &[u8], results: &mut [DartsResult]) -> usize {
        let max_results = results.len();
        let mut num_results = 0;

        let mut b = match self.get(0) {
            Some(unit) => unit.base,
            None => return 0,
        };

        for (i, &byte) in key.iter().enumerate() {
            // Check for a value at current position (before consuming the byte)
            // p = b (the base value points to the value node)
            let p = b as usize;
            if let Some(unit) = self.get(p) {
                // Check: b == array[p].check and array[p].base < 0
                if unit.check == b as u32 && unit.base < 0 && num_results < max_results {
                    results[num_results] = DartsResult {
                        value: -unit.base - 1,
                        length: i,
                    };
                    num_results += 1;
                }
            }

            // Transition to next state
            let p = (b as usize).wrapping_add(byte as usize).wrapping_add(1);

            match self.get(p) {
                // Check: b == array[p].check (parent's base equals child's check)
                Some(unit) if unit.check == b as u32 => {
                    b = unit.base;
                }
                _ => return num_results,
            }
        }

        // Check for a value at the final position
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
}
