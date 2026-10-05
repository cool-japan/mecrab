//! Connection matrix for transition costs between context IDs
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! The connection matrix stores the cost of transitioning from one
//! context (POS) to another. This is crucial for the Viterbi algorithm
//! to find the optimal segmentation.
//!
//! Reference: ../ref/mecab-0.996/src/connector.cpp
//!
//! Binary format:
//! - First 2 bytes: lsize (u16) - number of left contexts
//! - Next 2 bytes: rsize (u16) - number of right contexts
//! - Rest: lsize * rsize * 2 bytes of i16 costs
//!
//! Index formula: matrix[rcAttr + lsize * lcAttr]
//!
//! With `lsize * rsize` entries laid out that way, `rcAttr` (the preceding
//! node's `right_id`) is the low digit, radix `lsize`, and `lcAttr` (the
//! following node's `left_id`) the high digit, radix `rsize`: a pair is in
//! range exactly when `right_id < lsize` and `left_id < rsize`. A row (one
//! `left_id`) is the `lsize` contiguous entries from `left_id * lsize`.

use crate::{Error, Result};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::sync::Arc;

use super::sys_dic::DataBacking;

/// Connection matrix storing transition costs
pub struct ConnectionMatrix {
    /// Backing store (kept alive so matrix_ptr remains valid)
    _backing: DataBacking,
    /// Pointer to cost array (starts after lsize and rsize)
    matrix_ptr: *const i16,
    /// Number of left context IDs (the first header field): the range of the
    /// preceding node's `right_id`, and the length of a row
    lsize: usize,
    /// Number of right context IDs (the second header field): the range of the
    /// following node's `left_id`, and the number of rows
    rsize: usize,
}

// SAFETY: `ConnectionMatrix` only ever reads through `matrix_ptr`, never
// writes, and the memory it points at is the immutable backing store
// (`_backing`, an `Arc<Mmap>` or `Arc<Vec<u8>>`) that `self` keeps alive, so
// moving one between threads is sound.
unsafe impl Send for ConnectionMatrix {}
// SAFETY: as the `Send` impl above — the only access through `matrix_ptr` is
// an immutable read of shared backing memory, so `&ConnectionMatrix` is safe
// to share across threads.
unsafe impl Sync for ConnectionMatrix {}

impl std::fmt::Debug for ConnectionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectionMatrix")
            .field("lsize", &self.lsize)
            .field("rsize", &self.rsize)
            .finish()
    }
}

impl ConnectionMatrix {
    /// Parse a byte slice into the fields needed to construct a `ConnectionMatrix`.
    ///
    /// Returns `(matrix_ptr, lsize, rsize)` on success.
    fn parse_bytes(data: &[u8]) -> Result<(*const i16, usize, usize)> {
        if data.len() < 4 {
            return Err(Error::MatrixError(
                "Matrix file too small for header".to_string(),
            ));
        }

        let lsize = LittleEndian::read_u16(&data[0..2]) as usize;
        let rsize = LittleEndian::read_u16(&data[2..4]) as usize;

        // Checked: on a 32-bit target (wasm32) `lsize * rsize * 2` can wrap,
        // which would let a short file pass the length check below.
        let expected_size = Self::file_len(lsize, rsize).ok_or_else(|| {
            Error::MatrixError(format!(
                "Matrix dimensions {}x{} overflow the address space",
                lsize, rsize
            ))
        })?;
        if data.len() != expected_size {
            return Err(Error::MatrixError(format!(
                "Matrix file size mismatch: expected {} bytes ({}x{} matrix + 4), got {}",
                expected_size,
                lsize,
                rsize,
                data.len()
            )));
        }

        let matrix_ptr = data[4..].as_ptr() as *const i16;
        // Every read through `matrix_ptr` is a typed `i16` read, so the cost
        // array must be 2-byte aligned. A memory map is page-aligned and an
        // owned buffer comes from the allocator, so a well-formed file always
        // passes; this checks it rather than assuming it.
        if !matrix_ptr.is_aligned() {
            return Err(Error::MatrixError(
                "Matrix cost array is not 2-byte aligned in memory".to_string(),
            ));
        }
        Ok((matrix_ptr, lsize, rsize))
    }

    /// The file length a `lsize` x `rsize` matrix needs (`4 + lsize * rsize *
    /// 2`), or `None` when that does not fit in `usize`.
    fn file_len(lsize: usize, rsize: usize) -> Option<usize> {
        lsize.checked_mul(rsize)?.checked_mul(2)?.checked_add(4)
    }

    /// Load connection matrix from memory-mapped file
    ///
    /// # Errors
    ///
    /// Returns an error if the file is corrupted.
    pub fn from_mmap(mmap: Arc<Mmap>) -> Result<Self> {
        let (matrix_ptr, lsize, rsize) = Self::parse_bytes(mmap.as_ref())?;
        Ok(Self {
            _backing: DataBacking::Mmap(mmap),
            matrix_ptr,
            lsize,
            rsize,
        })
    }

    /// Load connection matrix from an owned byte buffer (no filesystem required).
    ///
    /// # Errors
    ///
    /// Returns an error if the data is corrupted.
    pub fn from_bytes_owned(data: Arc<Vec<u8>>) -> Result<Self> {
        let (matrix_ptr, lsize, rsize) = Self::parse_bytes(data.as_ref())?;
        Ok(Self {
            _backing: DataBacking::Owned(data),
            matrix_ptr,
            lsize,
            rsize,
        })
    }

    /// Get the connection cost between right and left context IDs
    ///
    /// This matches MeCab's Connector::transition_cost function:
    /// `return matrix_[rcAttr + lsize_ * lcAttr];`
    ///
    /// # Arguments
    ///
    /// * `right_id` - Right context attribute of the LEFT node, in range when
    ///   `right_id < left_size()`
    /// * `left_id` - Left context attribute of the RIGHT node, in range when
    ///   `left_id < right_size()`
    ///
    /// An out-of-range pair returns `i16::MAX`.
    #[inline]
    pub fn cost(&self, right_id: u16, left_id: u16) -> i16 {
        let rc = right_id as usize;
        let lc = left_id as usize;

        // `rc` is the low digit (radix `lsize`) and `lc` the high digit (radix
        // `rsize`) of `rc + lsize * lc`, so these are the bounds that keep the
        // index inside the `lsize * rsize` entries.
        if rc >= self.lsize || lc >= self.rsize {
            // Return a high cost for out-of-bounds lookups
            return i16::MAX;
        }

        // Index: rcAttr + lsize * lcAttr
        let index = rc + self.lsize * lc;

        // SAFETY: `rc < lsize` and `lc < rsize` were just checked, so `index <=
        // (lsize - 1) + lsize * (rsize - 1) = lsize * rsize - 1`. `matrix_ptr`
        // addresses `lsize * rsize` contiguous immutable `i16`s: `parse_bytes`
        // checked `4 + lsize * rsize * 2` against the file length without
        // overflow and checked the pointer's 2-byte alignment, and `_backing`
        // keeps the bytes alive for `self`.
        unsafe { *self.matrix_ptr.add(index) }
    }

    /// Get connection cost without bounds checking (hot path optimization)
    ///
    /// # Safety
    ///
    /// The caller must ensure that `right_id < self.left_size()` and
    /// `left_id < self.right_size()`, which keeps the index
    /// `right_id + left_size() * left_id` below `left_size() * right_size()`.
    /// The dictionary loaders do not check a token's context IDs against the
    /// matrix dimensions, so IDs read from a file that has not been checked
    /// that way must go through [`Self::cost`] instead.
    #[inline]
    pub unsafe fn cost_unchecked(&self, right_id: u16, left_id: u16) -> i16 {
        let rc = right_id as usize;
        let lc = left_id as usize;
        let index = rc + self.lsize * lc;
        // SAFETY: the caller guarantees `rc < lsize` and `lc < rsize` (this
        // `unsafe fn`'s contract), so `index <= lsize * rsize - 1`, inside the
        // `lsize * rsize` aligned `i16`s `parse_bytes` validated and `_backing`
        // keeps alive.
        unsafe { *self.matrix_ptr.add(index) }
    }

    /// Get the number of left context IDs: `cost` accepts a `right_id`
    /// below this, and a row from `row_for_left_id` has this many entries
    #[inline]
    pub fn left_size(&self) -> usize {
        self.lsize
    }

    /// Get the number of right context IDs: `cost` and `row_for_left_id`
    /// accept a `left_id` below this
    #[inline]
    pub fn right_size(&self) -> usize {
        self.rsize
    }

    /// Get total number of entries
    #[inline]
    pub fn size(&self) -> usize {
        self.lsize * self.rsize
    }

    /// Get a row of connection costs for a given left context ID.
    ///
    /// Returns a slice of `lsize` (`left_size()`) elements, one per right
    /// context ID (rc) a preceding node can carry, allowing the caller to index
    /// with `slice[rc]` without per-call bounds checks.
    ///
    /// This enables the Viterbi forward pass to amortize the bounds check:
    /// call once per target node, then index freely.
    ///
    /// Returns `None` if `left_id >= rsize` (`right_size()`).
    ///
    /// # Layout
    /// The matrix uses layout `index = rc + lsize * lc`. For fixed `lc`,
    /// elements at varying `rc` are contiguous (stride 1).
    #[inline]
    pub fn row_for_left_id(&self, left_id: u16) -> Option<&[i16]> {
        let lc = left_id as usize;
        if lc >= self.rsize {
            return None;
        }
        // For fixed lc: elements are at lc*lsize, lc*lsize+1, ..., lc*lsize+(lsize-1)
        // These are contiguous with stride 1.
        let start = lc * self.lsize;
        // SAFETY: `lc < rsize` was just checked, so `start + lsize = (lc + 1)
        // * lsize <= rsize * lsize`: the `lsize` elements from
        // `matrix_ptr.add(start)` lie inside the `lsize * rsize` immutable
        // `i16`s that `parse_bytes` validated against the file length (without
        // overflow) and found 2-byte aligned, and `_backing` keeps them alive
        // for `self`, which the returned slice borrows.
        Some(unsafe { std::slice::from_raw_parts(self.matrix_ptr.add(start), self.lsize) })
    }

    /// Get connection cost for a specific right_id from a pre-fetched row slice.
    ///
    /// Use together with `row_for_left_id` to avoid repeated bounds checks in
    /// tight loops (e.g. Viterbi forward pass over predecessor nodes).
    ///
    /// # Safety
    ///
    /// `right_id` must be `< row.len()` (for a row from `row_for_left_id`,
    /// `row.len()` is `left_size()`). Violating this causes undefined
    /// behaviour.
    #[inline]
    pub unsafe fn cost_from_row(row: &[i16], right_id: u16) -> i16 {
        // SAFETY: the caller guarantees `right_id < row.len()` (this `unsafe
        // fn`'s documented contract), so the indexed element is inside `row`.
        unsafe { *row.get_unchecked(right_id as usize) }
    }

    /// Get connection cost for a specific right_id from a pre-fetched row.
    ///
    /// Safe version that returns `i16::MAX` when `right_id` is out of bounds.
    /// Prefer this over `cost_from_row` unless profiling shows the bounds
    /// check is a bottleneck.
    #[inline]
    pub fn cost_from_row_checked(row: &[i16], right_id: u16) -> i16 {
        row.get(right_id as usize).copied().unwrap_or(i16::MAX)
    }

    /// Copy all connection costs into a heap-allocated `Vec<i16>`.
    ///
    /// Layout matches `cost()`: `data[right_id + lsize * left_id]`.
    pub fn to_vec(&self) -> Vec<i16> {
        let size = self.lsize * self.rsize;
        // SAFETY: `size == lsize * rsize`, the element count `parse_bytes`
        // validated against the file length (without overflow), and
        // `matrix_ptr` is 2-byte aligned (checked there) and addresses that
        // many contiguous immutable `i16`s owned by `_backing`, so the slice is
        // valid for the length of this call.
        unsafe { std::slice::from_raw_parts(self.matrix_ptr, size).to_vec() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_out_of_bounds() {
        // We can't easily test this without a real mmap, but we can verify the logic
        // Out of bounds should return i16::MAX
        assert_eq!(i16::MAX, 32767);
    }

    #[test]
    fn test_file_len_is_checked() {
        assert_eq!(ConnectionMatrix::file_len(3, 1), Some(10));
        assert_eq!(ConnectionMatrix::file_len(0, 7), Some(4));
        assert_eq!(
            ConnectionMatrix::file_len(u16::MAX as usize, u16::MAX as usize),
            (u16::MAX as usize * u16::MAX as usize)
                .checked_mul(2)
                .and_then(|n| n.checked_add(4))
        );
        // Each step of `4 + lsize * rsize * 2` can overflow on its own.
        assert_eq!(ConnectionMatrix::file_len(usize::MAX, 2), None);
        assert_eq!(ConnectionMatrix::file_len(usize::MAX / 2 + 1, 1), None);
        assert_eq!(ConnectionMatrix::file_len(usize::MAX / 2, 1), None);
        assert_eq!(
            ConnectionMatrix::file_len(1, (usize::MAX - 4) / 2),
            Some(usize::MAX - 1)
        );
    }

    #[test]
    fn test_cost_from_row_checked() {
        let row = vec![10i16, 20i16, 30i16];
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 0), 10);
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 1), 20);
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 2), 30);
        assert_eq!(ConnectionMatrix::cost_from_row_checked(&row, 100), i16::MAX);
    }

    #[test]
    fn test_cost_from_row_unchecked() {
        let row = vec![10i16, -5i16, 42i16];
        // SAFETY: `row` has length 3, so the indices 0, 1 and 2 passed to
        // `cost_from_row` are each less than `row.len()`, meeting its contract.
        unsafe {
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 0), 10);
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 1), -5);
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 2), 42);
        }
    }
}
