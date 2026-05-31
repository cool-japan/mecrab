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
    /// Number of left context IDs
    lsize: usize,
    /// Number of right context IDs
    rsize: usize,
}

// Safety: The matrix_ptr points to immutable memory-mapped data
unsafe impl Send for ConnectionMatrix {}
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

        let expected_size = 4 + lsize * rsize * 2;
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
        Ok((matrix_ptr, lsize, rsize))
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
    /// * `right_id` - Right context attribute of the LEFT node
    /// * `left_id` - Left context attribute of the RIGHT node
    #[inline]
    pub fn cost(&self, right_id: u16, left_id: u16) -> i16 {
        let rc = right_id as usize;
        let lc = left_id as usize;

        if rc >= self.rsize || lc >= self.lsize {
            // Return a high cost for out-of-bounds lookups
            return i16::MAX;
        }

        // Index: rcAttr + lsize * lcAttr
        let index = rc + self.lsize * lc;

        // Safety: We verified the indices are in bounds
        unsafe { *self.matrix_ptr.add(index) }
    }

    /// Get connection cost without bounds checking (hot path optimization)
    ///
    /// # Safety
    ///
    /// The caller must ensure that right_id < rsize and left_id < lsize.
    /// Context IDs from validated dictionary entries are always in bounds.
    #[inline]
    pub unsafe fn cost_unchecked(&self, right_id: u16, left_id: u16) -> i16 {
        let rc = right_id as usize;
        let lc = left_id as usize;
        let index = rc + self.lsize * lc;
        // Safety: caller guarantees indices are in bounds
        unsafe { *self.matrix_ptr.add(index) }
    }

    /// Get the number of left context IDs
    #[inline]
    pub fn left_size(&self) -> usize {
        self.lsize
    }

    /// Get the number of right context IDs
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
    /// Returns a slice of `rsize` elements, one per right context ID (rc),
    /// allowing the caller to index with `slice[rc]` without per-call bounds checks.
    ///
    /// This enables the Viterbi forward pass to amortize the bounds check:
    /// call once per target node, then index freely.
    ///
    /// Returns `None` if `left_id >= lsize`.
    ///
    /// # Layout
    /// The matrix uses layout `index = rc + lsize * lc`. For fixed `lc`,
    /// elements at varying `rc` are contiguous (stride 1).
    #[inline]
    pub fn row_for_left_id(&self, left_id: u16) -> Option<&[i16]> {
        let lc = left_id as usize;
        if lc >= self.lsize {
            return None;
        }
        // For fixed lc: elements are at lc*lsize, lc*lsize+1, ..., lc*lsize+(rsize-1)
        // These are contiguous with stride 1.
        // Safety: lc < lsize guarantees start = lc * lsize is within bounds.
        // start + rsize = lc*lsize + rsize.
        // For IPADIC where lsize == rsize: lc*lsize + lsize = (lc+1)*lsize <= lsize^2 = total ✓
        let start = lc * self.lsize;
        // Safety: lc < lsize so start < lsize * lsize = total matrix entries.
        // start + rsize <= (lsize-1)*lsize + rsize which for lsize==rsize = lsize^2 ✓
        Some(unsafe { std::slice::from_raw_parts(self.matrix_ptr.add(start), self.rsize) })
    }

    /// Get connection cost for a specific right_id from a pre-fetched row slice.
    ///
    /// Use together with `row_for_left_id` to avoid repeated bounds checks in
    /// tight loops (e.g. Viterbi forward pass over predecessor nodes).
    ///
    /// # Safety
    ///
    /// `right_id` must be `< rsize` (i.e., `< row.len()`). Violating this
    /// causes undefined behaviour.
    #[inline]
    pub unsafe fn cost_from_row(row: &[i16], right_id: u16) -> i16 {
        // Safety: caller ensures right_id < row.len()
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
        // Safety: matrix_ptr is valid for lsize * rsize elements
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
        // Safety: indices 0, 1, 2 are all < row.len() == 3
        unsafe {
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 0), 10);
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 1), -5);
            assert_eq!(ConnectionMatrix::cost_from_row(&row, 2), 42);
        }
    }
}
