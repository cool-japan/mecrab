//! Cached wrapper around `ConnectionMatrix` for hot-path optimization.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! # Design
//!
//! The Viterbi forward pass visits each lattice node and iterates over all
//! predecessor nodes. For a given target node the `left_id` is **fixed**,
//! so `row_for_left_id()` is called once per target but the same row is
//! reused for every predecessor.  A direct-mapped cache with
//! `CACHE_SLOTS = 256` entries (slot = `left_id & 0xFF`) eliminates even
//! the single bounds check that `row_for_left_id` performs on every access.
//!
//! # Thread safety
//!
//! `CachedMatrix` uses `UnsafeCell` for interior mutability.  The cache
//! writes are safe under the single-threaded constraint of the Viterbi
//! solver: each call to `solve()` creates its own lattice and processes
//! nodes sequentially.  Do **not** share a `CachedMatrix` between threads
//! without external synchronisation.

use std::cell::UnsafeCell;

use super::connection_matrix::ConnectionMatrix;

/// Number of direct-mapped cache slots.  Must be a power of two so that
/// `left_id & (CACHE_SLOTS - 1)` can be used as the slot index.
const CACHE_SLOTS: usize = 256;

/// One slot in the direct-mapped cache.
struct CacheEntry {
    /// The `left_id` whose row is cached here.
    left_id: u16,
    /// Whether this slot holds a valid entry.
    valid: bool,
    /// Pointer to the start of the row inside the memory-mapped matrix.
    row_ptr: *const i16,
    /// Length of the row (`== rsize` of the connection matrix).
    row_len: usize,
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            left_id: 0,
            valid: false,
            row_ptr: std::ptr::null(),
            row_len: 0,
        }
    }
}

// Safety: `row_ptr` always points into an `Arc<Mmap>` owned by the
// `ConnectionMatrix` that wraps `CachedMatrix`.  The mmap (and therefore
// the pointed-to memory) outlives every `CacheEntry`.
unsafe impl Send for CacheEntry {}
unsafe impl Sync for CacheEntry {}

/// Connection matrix with a direct-mapped row cache for hot-path use.
///
/// Each `left_id` maps to slot `left_id % CACHE_SLOTS`.  A cache hit
/// requires only a comparison and an unsafe slice reconstruction; a miss
/// delegates to `ConnectionMatrix::row_for_left_id` and writes the result
/// into the slot.
///
/// # Example
///
/// ```ignore
/// // In the Viterbi forward pass:
/// let cached = CachedMatrix::new(matrix);
/// for node in lattice.nodes() {
///     // Single cache lookup per target node:
///     if let Some(row) = cached.row_for_left_id(node.left_id) {
///         for pred in lattice.predecessors(node) {
///             // No further bounds checks needed:
///             let cost = ConnectionMatrix::cost_from_row_checked(row, pred.right_id);
///             // … update Viterbi table …
///         }
///     }
/// }
/// ```
pub struct CachedMatrix {
    /// The underlying (immutable) matrix.
    inner: ConnectionMatrix,
    /// Direct-mapped row cache with interior mutability.
    ///
    /// Interior mutability is needed so that `row_for_left_id` can update
    /// the cache through a shared (`&self`) reference, matching the borrow
    /// requirements of the Viterbi solver which also needs `&self` access
    /// to the lattice.
    cache: UnsafeCell<Vec<CacheEntry>>,
}

// Safety: See module-level doc.  `CachedMatrix` must only be used from
// a single thread.  The `Sync` impl is required so it can live inside
// structures that are `Send + Sync`; callers must uphold the single-thread
// invariant.
unsafe impl Sync for CachedMatrix {}

// Safety: `CachedMatrix` owns its `ConnectionMatrix` (which is Send) and
// the `UnsafeCell<Vec<CacheEntry>>` (which is Send through the CacheEntry
// Send impl above).
unsafe impl Send for CachedMatrix {}

impl std::fmt::Debug for CachedMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedMatrix")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl CachedMatrix {
    /// Wrap a `ConnectionMatrix` with a fresh (empty) cache.
    pub fn new(inner: ConnectionMatrix) -> Self {
        let cache = (0..CACHE_SLOTS).map(|_| CacheEntry::default()).collect();
        Self {
            inner,
            cache: UnsafeCell::new(cache),
        }
    }

    /// Look up the row for `left_id`, populating the cache on a miss.
    ///
    /// Returns `None` if `left_id >= lsize`.
    ///
    /// The returned slice is valid for the lifetime of `self` because the
    /// underlying mmap is kept alive by the `ConnectionMatrix`.
    #[inline]
    pub fn row_for_left_id(&self, left_id: u16) -> Option<&[i16]> {
        let slot = (left_id as usize) & (CACHE_SLOTS - 1);

        // Safety: single-threaded use is guaranteed by the caller (see
        // module-level doc).  No other code path aliases the same slot
        // concurrently.
        let cache = unsafe { &mut *self.cache.get() };
        let entry = &mut cache[slot];

        if entry.valid && entry.left_id == left_id {
            // Cache hit — reconstruct the slice from the stored raw pointer.
            // Safety: `row_ptr` was obtained from `ConnectionMatrix::row_for_left_id`
            // which guarantees the pointer remains valid for `self`'s lifetime.
            return Some(unsafe { std::slice::from_raw_parts(entry.row_ptr, entry.row_len) });
        }

        // Cache miss — delegate to the inner matrix and store the result.
        let row = self.inner.row_for_left_id(left_id)?;

        *entry = CacheEntry {
            left_id,
            valid: true,
            row_ptr: row.as_ptr(),
            row_len: row.len(),
        };

        Some(row)
    }

    /// Invalidate the cache slot for the given `left_id`.
    ///
    /// This is only needed if you replace the underlying `ConnectionMatrix`
    /// (hot-swap scenario).  Under normal operation the matrix is immutable
    /// and invalidation is unnecessary.
    #[inline]
    pub fn invalidate(&self, left_id: u16) {
        let slot = (left_id as usize) & (CACHE_SLOTS - 1);
        // Safety: same single-thread invariant as `row_for_left_id`.
        let cache = unsafe { &mut *self.cache.get() };
        cache[slot].valid = false;
    }

    /// Invalidate all cache entries.
    pub fn invalidate_all(&self) {
        // Safety: same single-thread invariant as `row_for_left_id`.
        let cache = unsafe { &mut *self.cache.get() };
        for entry in cache.iter_mut() {
            entry.valid = false;
        }
    }

    /// Delegate: get connection cost using the standard (uncached) path.
    ///
    /// Use `row_for_left_id` + `ConnectionMatrix::cost_from_row_checked` for
    /// the hot path; use this method only in contexts where the row is not
    /// reused multiple times.
    #[inline]
    pub fn cost(&self, right_id: u16, left_id: u16) -> i16 {
        self.inner.cost(right_id, left_id)
    }

    /// Get a reference to the underlying `ConnectionMatrix`.
    #[inline]
    pub fn inner(&self) -> &ConnectionMatrix {
        &self.inner
    }

    /// Consume the wrapper, returning the inner `ConnectionMatrix`.
    pub fn into_inner(self) -> ConnectionMatrix {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify `cost_from_row_checked` (re-exported convenience test).
    #[test]
    fn test_cost_from_row_checked_via_cached() {
        let row = vec![10i16, 20i16, -5i16, 100i16];
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 0), 10);
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 2), -5);
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 99), i16::MAX);
    }

    /// Verify that the slot formula covers all IDs in `0..256` without collision.
    #[test]
    fn test_direct_mapped_cache_slot_no_collision_low_ids() {
        let slots: Vec<usize> = (0u16..256)
            .map(|id| (id as usize) & (CACHE_SLOTS - 1))
            .collect();
        assert_eq!(slots[0], 0);
        assert_eq!(slots[127], 127);
        assert_eq!(slots[255], 255);
        // Every low ID maps to a unique slot.
        let mut seen = vec![false; CACHE_SLOTS];
        for s in &slots {
            assert!(!seen[*s], "Collision detected at slot {s}");
            seen[*s] = true;
        }
    }

    /// Verify that IDs that differ by exactly `CACHE_SLOTS` map to the same slot
    /// (expected direct-mapped collision behaviour).
    #[test]
    fn test_direct_mapped_cache_collision_pattern() {
        // left_id 0 and left_id 256 must share slot 0.
        // 0 maps to slot 0 by definition.
        let slot_256: usize = 0x100_usize & (CACHE_SLOTS - 1);
        assert_eq!(slot_256, 0, "left_id 256 must map to slot 0");

        // left_id 1 and left_id 257 must share slot 1.
        let slot_1: usize = 1usize & (CACHE_SLOTS - 1);
        let slot_257: usize = 0x101_usize & (CACHE_SLOTS - 1);
        assert_eq!(slot_1, slot_257);
    }
}
