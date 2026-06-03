//! Character property table (char.bin) writer for MeCab-compatible dictionaries.
//!
//! Binary format (must match `mecrab/src/dict/char_def.rs`):
//! ```text
//! [u32 csize LE]
//! [csize × 32-byte NUL-padded category name fields]
//! [0xFFFF × u32 CharInfo values LE]
//! ```
//! Total = `4 + csize*32 + 0xFFFF*4` bytes.
//!
//! CharInfo bit layout (from `char_def.rs:69-115`):
//! - bits  0..18 : type bitmask (18 bits)
//! - bits 18..26 : default_type (8 bits, category id)
//! - bits 26..30 : length (4 bits)
//! - bit  30     : group flag
//! - bit  31     : invoke flag
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use byteorder::{LittleEndian, WriteBytesExt};

/// Number of code-point entries in the CharInfo lookup table.
///
/// Matches `CharDef::TABLE_SIZE = 0xFFFF = 65535` in the mecrab core.
pub const CHARINFO_TABLE_SIZE: usize = 0xFFFF;

/// A contiguous range of Unicode code points to assign a single CharInfo.
#[derive(Debug, Clone)]
pub struct CharRange {
    /// First code point in the range (inclusive)
    pub lo: u32,
    /// Last code point in the range (inclusive)
    pub hi: u32,
    /// Packed CharInfo value for every code point in `lo..=hi`
    pub info: u32,
}

/// Pack a CharInfo `u32` value from its components.
///
/// Bit layout (must match `char_def.rs:88-115`):
/// - bits  0..18 : `type_mask` (18 bits, must include the bit for `default_type`)
/// - bits 18..26 : `default_type` (category id, u8)
/// - bits 26..30 : `length` (4 bits)
/// - bit  30     : `group` (1 = group consecutive same-category chars)
/// - bit  31     : `invoke` (1 = invoke unknown-word processing)
///
/// # Arguments
/// * `type_mask`    - Bitmask of categories this code point belongs to (18 bits)
/// * `default_type` - Primary category id (u8)
/// * `length`       - Max grouping length in characters (4 bits)
/// * `group`        - Whether to group consecutive characters of this category
/// * `invoke`       - Whether to invoke unknown-word processing for this category
#[must_use]
pub fn pack_char_info(
    type_mask: u32,
    default_type: u8,
    length: u8,
    group: bool,
    invoke: bool,
) -> u32 {
    (type_mask & 0x3_FFFF)
        | ((default_type as u32) << 18)
        | (((length as u32) & 0xF) << 26)
        | ((group as u32) << 30)
        | ((invoke as u32) << 31)
}

/// Build the char.bin byte buffer.
///
/// The output is exactly `4 + categories.len()*32 + CHARINFO_TABLE_SIZE*4` bytes.
/// The reader (`CharDef::from_bytes_owned`) will reject any other size.
///
/// # Arguments
/// * `categories` - Category names in enum-id order (index 0 = DEFAULT, 1 = SPACE, …).
///   The order must match `CharCategory` in `char_def.rs`.
/// * `ranges` - Code-point ranges to assign. Ranges are applied in order;
///   later ranges overwrite earlier ones. Code points not covered
///   by any range receive `info = 0` (DEFAULT, no group/invoke).
///
/// # Errors
///
/// Returns an error if the byte buffer cannot be constructed.
pub fn build_char_bytes(categories: &[&str], ranges: &[CharRange]) -> crate::Result<Vec<u8>> {
    let csize = categories.len();
    let total = 4 + csize * 32 + CHARINFO_TABLE_SIZE * 4;
    let mut buf: Vec<u8> = Vec::with_capacity(total);

    // Write csize
    buf.write_u32::<LittleEndian>(csize as u32)?;

    // Write category name fields (32 bytes each, NUL-padded)
    for &name in categories {
        let name_bytes = name.as_bytes();
        let mut field = [0u8; 32];
        let copy_len = name_bytes.len().min(31);
        field[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        buf.extend_from_slice(&field);
    }

    // Fill the CharInfo table (65535 entries, default = 0)
    let mut table = vec![0u32; CHARINFO_TABLE_SIZE];

    // Apply ranges in order (later overwrites earlier)
    for range in ranges {
        let lo = range.lo as usize;
        let hi = (range.hi as usize).min(CHARINFO_TABLE_SIZE - 1);
        if lo < CHARINFO_TABLE_SIZE {
            for entry in table.iter_mut().skip(lo).take(hi + 1 - lo) {
                *entry = range.info;
            }
        }
    }

    // Write table entries
    for &info in &table {
        buf.write_u32::<LittleEndian>(info)?;
    }

    debug_assert_eq!(buf.len(), total, "char.bin size mismatch");
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_char_info_bits() {
        // type_mask=2 (bit 1), default_type=1, length=3, group=true, invoke=false
        let packed = pack_char_info(2, 1, 3, true, false);
        // type_mask (bits 0..18): 2
        assert_eq!(packed & 0x3_FFFF, 2);
        // default_type (bits 18..26): 1
        assert_eq!((packed >> 18) & 0xFF, 1);
        // length (bits 26..30): 3
        assert_eq!((packed >> 26) & 0xF, 3);
        // group (bit 30): 1
        assert_eq!((packed >> 30) & 1, 1);
        // invoke (bit 31): 0
        assert_eq!((packed >> 31) & 1, 0);
    }

    #[test]
    fn test_build_char_bytes_size() {
        let cats = ["DEFAULT", "SPACE", "KANJI"];
        let ranges: Vec<CharRange> = Vec::new();
        let bytes = build_char_bytes(&cats, &ranges).unwrap();
        let expected = 4 + 3 * 32 + CHARINFO_TABLE_SIZE * 4;
        assert_eq!(
            bytes.len(),
            expected,
            "char.bin must be exactly {expected} bytes"
        );
    }

    #[test]
    fn test_build_char_bytes_csize_field() {
        let cats = ["DEFAULT", "SPACE"];
        let bytes = build_char_bytes(&cats, &[]).unwrap();
        let csize = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(csize, 2);
    }

    #[test]
    fn test_range_application_hiragana() {
        let cats = [
            "DEFAULT", "SPACE", "KANJI", "SYMBOL", "NUMERIC", "ALPHA", "HIRAGANA",
        ];
        // Assign Hiragana range: U+3041..=U+3096, id=6
        let hiragana_info = pack_char_info(1 << 6, 6, 0, true, true);
        let ranges = vec![CharRange {
            lo: 0x3041,
            hi: 0x3096,
            info: hiragana_info,
        }];
        let bytes = build_char_bytes(&cats, &ranges).unwrap();

        // 'あ' = U+3042: table[0x3042]
        // offset into table = 4 + 7*32 + 0x3042 * 4
        let table_start = 4 + cats.len() * 32;
        let idx = table_start + 0x3042 * 4;
        let info = u32::from_le_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
        assert_eq!(info, hiragana_info, "'あ' CharInfo should match Hiragana");
        // group bit must be set
        assert_eq!((info >> 30) & 1, 1, "group bit must be set for Hiragana");
    }

    #[test]
    fn test_charinfo_table_size_constant() {
        assert_eq!(CHARINFO_TABLE_SIZE, 0xFFFF);
        assert_eq!(CHARINFO_TABLE_SIZE, 65535);
    }
}
