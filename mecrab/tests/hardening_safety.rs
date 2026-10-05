//! Regression tests for the memory-safety preconditions of the dictionary
//! loaders, each exercised through the safe public API:
//!
//! - a connection matrix answers by its flat layout `right_id + lsize *
//!   left_id`, in range exactly when `right_id < lsize` and `left_id < rsize`,
//!   and never reads past its buffer, square or not;
//! - a dictionary section the loaders read as wider values (the token array,
//!   the double array) must be aligned for them, and a misaligned one is
//!   rejected rather than read;
//! - a vector-store header whose `vocab_size * dim * element_size` overflows
//!   `usize` is rejected rather than accepted with a bogus size.

use std::sync::Arc;

use mecrab::TrainingMatrix;
use mecrab::dict::{ConnectionMatrix, DoubleArrayTrie, SysDic, Token};
use mecrab::vectors::VectorStore;

const DICTIONARY_MAGIC_ID: u32 = 0xef71_8f77;
const DIC_VERSION: u32 = 102;
const HEADER_SIZE: usize = 72;

/// A copy of `bytes` in a buffer whose start is `align`-aligned.
///
/// A `Vec<u8>` is only guaranteed byte alignment, and the loaders reject a
/// buffer whose sections are not aligned for the values read from them. Every
/// system allocator returns at least 8-byte-aligned blocks, so the first copy
/// is aligned there; an allocator that places byte buffers anywhere (Miri's
/// does) may need a few tries, and the earlier copies are kept so it cannot
/// hand the same block back.
fn aligned_copy(bytes: &[u8], align: usize) -> Arc<Vec<u8>> {
    let mut tries = Vec::new();
    for _ in 0..256 {
        let copy = bytes.to_vec();
        if copy.as_ptr().align_offset(align) == 0 {
            return Arc::new(copy);
        }
        tries.push(copy);
    }
    panic!("no {align}-aligned buffer in 256 allocations");
}

fn matrix(lsize: u16, rsize: u16, costs: &[i16]) -> ConnectionMatrix {
    assert_eq!(costs.len(), usize::from(lsize) * usize::from(rsize));
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&lsize.to_le_bytes());
    bytes.extend_from_slice(&rsize.to_le_bytes());
    for cost in costs {
        bytes.extend_from_slice(&cost.to_le_bytes());
    }
    // The costs start 4 bytes in and are read as `i16`.
    ConnectionMatrix::from_bytes_owned(aligned_copy(&bytes, 2)).expect("matrix should parse")
}

/// Checks every pair and every row of `m` against the flat layout of `costs`.
///
/// In-range pairs are checked first, so a matrix that answers one of them
/// wrongly fails before any out-of-range pair is asked.
fn assert_flat_layout(m: &ConnectionMatrix, lsize: u16, rsize: u16, costs: &[i16]) {
    assert_eq!(m.left_size(), usize::from(lsize));
    assert_eq!(m.right_size(), usize::from(rsize));
    assert_eq!(m.size(), costs.len());
    for left_id in 0..rsize {
        for right_id in 0..lsize {
            let expected = costs[usize::from(right_id) + usize::from(lsize) * usize::from(left_id)];
            assert_eq!(
                m.cost(right_id, left_id),
                expected,
                "cost({right_id}, {left_id}) on a {lsize}x{rsize} matrix"
            );
            // SAFETY: the loops keep `right_id < lsize == m.left_size()` and
            // `left_id < rsize == m.right_size()`, which is
            // `cost_unchecked`'s contract.
            let unchecked = unsafe { m.cost_unchecked(right_id, left_id) };
            assert_eq!(unchecked, expected);
        }
        let start = usize::from(left_id) * usize::from(lsize);
        assert_eq!(
            m.row_for_left_id(left_id),
            Some(&costs[start..start + usize::from(lsize)]),
            "row_for_left_id({left_id}) on a {lsize}x{rsize} matrix"
        );
    }
    // Every pair outside the bounds, including those whose flat index would
    // still fall inside the buffer, answers the sentinel; no row exists past
    // `rsize`.
    for left_id in 0..rsize + 2 {
        for right_id in 0..lsize + 2 {
            if right_id >= lsize || left_id >= rsize {
                assert_eq!(
                    m.cost(right_id, left_id),
                    i16::MAX,
                    "cost({right_id}, {left_id}) on a {lsize}x{rsize} matrix"
                );
            }
        }
    }
    for left_id in rsize..rsize + 2 {
        assert_eq!(m.row_for_left_id(left_id), None);
    }
    // `TrainingMatrix` copies the matrix and must answer every pair alike.
    let training = TrainingMatrix::from_connection_matrix(m);
    for left_id in 0..rsize + 2 {
        for right_id in 0..lsize + 2 {
            assert_eq!(training.cost(right_id, left_id), m.cost(right_id, left_id));
        }
    }
}

/// A `lsize = 3`, `rsize = 1` matrix: `right_id` ranges over `0..3` and
/// `left_id` over `0..1`, so the three entries are `(0, 0)`, `(1, 0)` and
/// `(2, 0)`, and the one row is all three. `(0, 2)` is out of range (its flat
/// index 6 is past the three entries) and must answer the sentinel without a
/// read.
#[test]
fn a_three_by_one_matrix_follows_the_flat_layout() {
    let costs = [10i16, 20, 30];
    let m = matrix(3, 1, &costs);
    assert_flat_layout(&m, 3, 1, &costs);
}

/// A `lsize = 1`, `rsize = 3` matrix: `right_id` ranges over `0..1` and
/// `left_id` over `0..3`, so the entries are `(0, 0)`, `(0, 1)` and `(0, 2)`,
/// and each of the three rows is one entry. `(1, 0)` is out of range even
/// though its flat index 1 is inside the buffer.
#[test]
fn a_one_by_three_matrix_follows_the_flat_layout() {
    let costs = [10i16, 20, 30];
    let m = matrix(1, 3, &costs);
    assert_flat_layout(&m, 1, 3, &costs);
}

/// On a square matrix (every dictionary the crate ships or builds) the bounds
/// are the same either way round; every pair and row matches the layout.
#[test]
fn a_square_matrix_follows_the_flat_layout() {
    let costs: Vec<i16> = (0..9).map(|i| 100 + i).collect();
    let m = matrix(3, 3, &costs);
    assert_flat_layout(&m, 3, 3, &costs);
}

/// A `sys.dic` image whose header sizes are `da`, one 16-byte token and
/// `feature` bytes, with the token's `left_id` set to 7; the magic number
/// encodes the image length, as the loader requires.
fn sysdic_image(da: usize, feature: usize) -> Vec<u8> {
    let total = HEADER_SIZE + da + Token::SIZE + feature;
    let mut bytes = vec![0u8; total];
    let header = [
        (total as u32) ^ DICTIONARY_MAGIC_ID,
        DIC_VERSION,
        0,
        0,
        0,
        0,
        da as u32,
        Token::SIZE as u32,
        feature as u32,
    ];
    for (i, field) in header.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&field.to_le_bytes());
    }
    bytes[HEADER_SIZE + da..HEADER_SIZE + da + 2].copy_from_slice(&7u16.to_le_bytes());
    bytes
}

/// The token section starts at `HEADER_SIZE + da_size`. A header whose
/// `da_size` leaves it misaligned for `Token` must be rejected: `token_at`
/// hands out `&Token`, and a misaligned reference is undefined behaviour.
/// (Red on the unpatched crate: the image loads, and `token_at(0)` is the
/// misaligned reference.) An aligned section still loads and reads.
#[test]
fn a_misaligned_token_section_is_rejected() {
    assert_eq!(std::mem::align_of::<Token>(), 4);
    // In a 4-aligned buffer the token section at `HEADER_SIZE + da` (72 + 1,
    // 2 or 3) is misaligned for every `da` below.
    for da in 1..=3 {
        let bytes = aligned_copy(&sysdic_image(da, 4 - da), 4);
        assert!(
            SysDic::from_bytes_owned(bytes).is_err(),
            "a token section at da_size = {da} is misaligned and must be rejected"
        );
    }
    // HEADER_SIZE + 8 is a multiple of 4, so an aligned buffer gives an
    // aligned token section, which loads and reads as before.
    let aligned = aligned_copy(&sysdic_image(8, 4), std::mem::align_of::<Token>());
    let dic = SysDic::from_bytes_owned(aligned).expect("aligned image loads");
    assert_eq!(dic.token_count(), 1);
    assert_eq!(dic.token_at(0).map(|t| t.left_id), Some(7));
}

/// `DoubleArrayTrie::from_bytes` keeps a pointer it reads whole 8-byte units
/// through (align 4); a non-empty array that does not start 4-byte aligned
/// must be rejected. (Red on the unpatched crate, which accepts it.)
#[test]
fn a_misaligned_double_array_is_rejected() {
    let bytes = [0u8; 32];
    let aligned = bytes.as_ptr().align_offset(4);
    assert!(aligned + 1 + 8 <= bytes.len());
    // SAFETY: `bytes` is a local array that outlives every trie made here
    // (each is dropped at the end of its statement) and is never mutated.
    unsafe {
        assert!(DoubleArrayTrie::from_bytes(&bytes[aligned..], 8).is_ok());
        assert!(DoubleArrayTrie::from_bytes(&bytes[aligned + 1..], 8).is_err());
        // An empty array is never read, so its start is not checked.
        assert!(DoubleArrayTrie::from_bytes(&bytes[aligned + 1..], 0).is_ok());
    }
}

/// A 32-byte MCV1 header declaring `vocab_size = dim = 2^31` (f32) would make
/// `vocab_size * dim * element_size` wrap to 0 in a build without overflow
/// checks, so the old equality `32 == 32 + 0` accepted a header-only file and
/// then handed out slices over data that is not there. The size validation must
/// now reject it. (Red on the unpatched crate: it panics on the multiply in a
/// debug build and accepts the file in a release build.)
#[test]
fn vector_header_size_overflow_is_rejected() {
    const MAGIC: u32 = 0x3143_564D; // "MCV1"
    let mut header = Vec::new();
    header.extend_from_slice(&MAGIC.to_le_bytes());
    header.extend_from_slice(&(1u32 << 31).to_le_bytes()); // vocab_size
    header.extend_from_slice(&(1u32 << 31).to_le_bytes()); // dim
    header.extend_from_slice(&0u32.to_le_bytes()); // data_type = f32
    header.extend_from_slice(&[0u8; 16]); // reserved
    assert_eq!(header.len(), 32);

    let dir = std::env::temp_dir();
    let path = dir.join(format!("mecrab_overflow_{}.mcv", std::process::id()));
    std::fs::write(&path, &header).expect("write temp vector file");
    let result = VectorStore::from_file(&path);
    let _ = std::fs::remove_file(&path);
    assert!(
        result.is_err(),
        "a header whose vocab_size * dim overflows usize must be rejected"
    );
}
