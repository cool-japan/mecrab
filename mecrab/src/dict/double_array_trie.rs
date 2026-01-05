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
