//! Connection matrix (matrix.bin) writer for MeCab-compatible dictionaries.
//!
//! The binary format is:
//! ```text
//! [u16 lsize LE][u16 rsize LE][i16 costs LE, row-major]
//! ```
//! Total = `4 + lsize * rsize * 2` bytes.
//!
//! Index formula: `cost(right_id, left_id) = costs[right_id + lsize * left_id]`
//! (matches `connection_matrix.rs:129` — `index = rc + lsize * lc`).
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::{BuildError, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::path::Path;

/// Build a MeCab-compatible matrix.bin byte buffer.
///
/// # Arguments
/// * `lsize` - Number of left context IDs
/// * `rsize` - Number of right context IDs
/// * `costs` - Flat cost array of length `lsize * rsize`, in row-major order.
///   `costs[right_id + lsize * left_id]` = connection cost.
///
/// # Errors
///
/// Returns an error if `costs.len() != lsize as usize * rsize as usize`.
pub fn build_matrix_bytes(lsize: u16, rsize: u16, costs: &[i16]) -> Result<Vec<u8>> {
    let expected_len = lsize as usize * rsize as usize;
    if costs.len() != expected_len {
        return Err(BuildError::InvalidInput(format!(
            "costs.len() = {} but lsize({}) * rsize({}) = {}",
            costs.len(),
            lsize,
            rsize,
            expected_len,
        )));
    }

    let total_bytes = 4 + expected_len * 2;
    let mut buf: Vec<u8> = Vec::with_capacity(total_bytes);

    buf.write_u16::<LittleEndian>(lsize)?;
    buf.write_u16::<LittleEndian>(rsize)?;

    for &cost in costs {
        buf.write_i16::<LittleEndian>(cost)?;
    }

    Ok(buf)
}

/// Set a specific cost value in a flat costs slice.
///
/// The index formula mirrors `connection_matrix.rs:129`:
/// `index = right_id + lsize * left_id`
///
/// # Arguments
/// * `costs`    - Mutable flat costs slice (`lsize * rsize` entries)
/// * `lsize`    - Number of left context IDs (= row stride)
/// * `right_id` - Right context attribute of the left node
/// * `left_id`  - Left context attribute of the right node
/// * `value`    - The i16 cost to store
///
/// # Panics
///
/// Panics if `right_id + lsize * left_id >= costs.len()`.
pub fn set_cost(
    costs: &mut [i16],
    lsize: usize,
    right_id: usize,
    left_id: usize,
    value: i16,
) {
    costs[right_id + lsize * left_id] = value;
}

/// Write a MeCab-compatible matrix.bin file.
///
/// # Errors
///
/// Returns an error if the costs slice is the wrong length or if the file
/// cannot be written.
pub fn write_matrix(lsize: u16, rsize: u16, costs: &[i16], output: &Path) -> Result<()> {
    let bytes = build_matrix_bytes(lsize, rsize, costs)?;
    std::fs::write(output, &bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_matrix_bytes_size() {
        let lsize: u16 = 4;
        let rsize: u16 = 4;
        let costs = vec![0i16; 16];
        let bytes = build_matrix_bytes(lsize, rsize, &costs).unwrap();
        // Expected: 4 header + 16 * 2 = 36 bytes
        assert_eq!(bytes.len(), 4 + 16 * 2);
    }

    #[test]
    fn test_build_matrix_bytes_header() {
        let costs = vec![0i16; 6];
        let bytes = build_matrix_bytes(2, 3, &costs).unwrap();
        let lsize = u16::from_le_bytes([bytes[0], bytes[1]]);
        let rsize = u16::from_le_bytes([bytes[2], bytes[3]]);
        assert_eq!(lsize, 2);
        assert_eq!(rsize, 3);
    }

    #[test]
    fn test_set_cost_indexing() {
        let mut costs = vec![0i16; 4 * 4];
        // right_id=1, left_id=2 → index = 1 + 4*2 = 9
        set_cost(&mut costs, 4, 1, 2, 42);
        assert_eq!(costs[9], 42);
    }

    #[test]
    fn test_build_matrix_wrong_len_errors() {
        let costs = vec![0i16; 5]; // wrong: should be 4 for 2x2
        let result = build_matrix_bytes(2, 2, &costs);
        assert!(result.is_err());
    }

    #[test]
    fn test_roundtrip_cost_values() {
        let lsize: u16 = 8;
        let rsize: u16 = 8;
        let mut costs = vec![1000i16; 64];
        set_cost(&mut costs, 8, 0, 1, 0);    // BOS → noun
        set_cost(&mut costs, 8, 1, 0, 0);    // noun → EOS
        set_cost(&mut costs, 8, 1, 2, 0);    // noun → particle
        set_cost(&mut costs, 8, 2, 1, 0);    // particle → noun
        let bytes = build_matrix_bytes(lsize, rsize, &costs).unwrap();

        // Verify size
        assert_eq!(bytes.len(), 4 + 64 * 2);

        // Verify a specific cost by reading back
        // cost(right_id=0, left_id=1) at index 0 + 8*1 = 8 → bytes at 4 + 8*2 = 20
        let idx = 4 + 8 * 2;
        let cost_val = i16::from_le_bytes([bytes[idx], bytes[idx + 1]]);
        assert_eq!(cost_val, 0, "cost(right=0, left=1) should be 0");
    }
}
