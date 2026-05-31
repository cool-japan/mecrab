//! Packed-blob dictionary loading for the WASM bindings.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module implements the binary header parser for the MeCrab packed
//! dictionary blob that JavaScript callers supply to `loadDictionary`.
//!
//! # Packed dictionary blob format
//!
//! JavaScript callers must supply a single `Uint8Array` (packed blob) that
//! concatenates the four MeCab binary files with a 36-byte header:
//!
//! ```text
//! Offset  Size  Field
//! ──────  ────  ─────────────────────────────────────────
//!      0     4  magic = 0x4D434142  ("MCAB" in LE)
//!      4     4  offset of sys.dic data  (from blob start)
//!      8     4  length of sys.dic data
//!     12     4  offset of matrix.bin data
//!     16     4  length of matrix.bin data
//!     20     4  offset of char.bin data
//!     24     4  length of char.bin data
//!     28     4  offset of unk.dic data
//!     32     4  length of unk.dic data
//!     36     …  data sections (in any order; offsets above must match)
//! ```

/// Magic constant identifying a MeCrab packed dictionary blob (`"MCAB"` in LE u32).
pub(super) const BLOB_MAGIC: u32 = 0x4D43_4142;

/// Byte size of the packed-blob header (1 magic + 4×2 offset/length pairs = 9 u32 = 36 bytes).
pub(super) const HEADER_BYTES: usize = 36;

/// Parse the packed blob and return a fully initialised [`crate::dict::Dictionary`].
///
/// Returns `Ok(dict)` on success, or `Err(description)` on any header or
/// bounds violation.
pub(super) fn parse_blob(data: &[u8]) -> Result<crate::dict::Dictionary, String> {
    if data.len() < HEADER_BYTES {
        return Err(format!(
            "Blob too small: need at least {} bytes for header, got {}",
            HEADER_BYTES,
            data.len()
        ));
    }

    // Read magic (LE u32 at offset 0).
    let magic = u32::from_le_bytes(
        data[0..4]
            .try_into()
            .map_err(|_| "Failed to read magic bytes".to_string())?,
    );
    if magic != BLOB_MAGIC {
        return Err(format!(
            "Invalid magic: expected 0x{:08X}, got 0x{:08X}",
            BLOB_MAGIC, magic
        ));
    }

    // Read one (offset, length) pair from the header at `base`.
    let read_section = |base: usize| -> Result<(usize, usize), String> {
        let off_bytes: [u8; 4] = data[base..base + 4]
            .try_into()
            .map_err(|_| format!("Failed to read offset at header[{}]", base))?;
        let len_bytes: [u8; 4] = data[base + 4..base + 8]
            .try_into()
            .map_err(|_| format!("Failed to read length at header[{}]", base + 4))?;
        Ok((
            u32::from_le_bytes(off_bytes) as usize,
            u32::from_le_bytes(len_bytes) as usize,
        ))
    };

    let (sys_off, sys_len) = read_section(4)?;
    let (mat_off, mat_len) = read_section(12)?;
    let (chr_off, chr_len) = read_section(20)?;
    let (unk_off, unk_len) = read_section(28)?;

    // Validate that every section lies within the blob.
    let check_bounds = |label: &str, off: usize, len: usize| -> Result<(), String> {
        let end = off
            .checked_add(len)
            .ok_or_else(|| format!("{} section offset+length overflows usize", label))?;
        if end > data.len() {
            return Err(format!(
                "{} section [{}..{}] out of blob bounds (blob len={})",
                label,
                off,
                end,
                data.len()
            ));
        }
        Ok(())
    };
    check_bounds("sys.dic", sys_off, sys_len)?;
    check_bounds("matrix.bin", mat_off, mat_len)?;
    check_bounds("char.bin", chr_off, chr_len)?;
    check_bounds("unk.dic", unk_off, unk_len)?;

    crate::dict::Dictionary::from_bytes(
        &data[sys_off..sys_off + sys_len],
        &data[mat_off..mat_off + mat_len],
        &data[chr_off..chr_off + chr_len],
        &data[unk_off..unk_off + unk_len],
    )
    .map_err(|e| e.to_string())
}
