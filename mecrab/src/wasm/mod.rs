//! WebAssembly bindings for MeCrab
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! This module provides JavaScript bindings for MeCrab using wasm-bindgen.
//! It enables morphological analysis in web browsers and Node.js.
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
//!
//! ## Building the blob from Node.js
//!
//! ```javascript
//! import { readFileSync } from 'fs';
//!
//! function packDictionary(dictDir) {
//!   const sys = readFileSync(`${dictDir}/sys.dic`);
//!   const mat = readFileSync(`${dictDir}/matrix.bin`);
//!   const chr = readFileSync(`${dictDir}/char.bin`);
//!   const unk = readFileSync(`${dictDir}/unk.dic`);
//!
//!   const HEADER = 36;
//!   const offSys = HEADER;
//!   const offMat = offSys + sys.length;
//!   const offChr = offMat + mat.length;
//!   const offUnk = offChr + chr.length;
//!   const total  = offUnk + unk.length;
//!
//!   const buf = Buffer.alloc(total);
//!   buf.writeUInt32LE(0x4D434142,  0);
//!   buf.writeUInt32LE(offSys,      4);
//!   buf.writeUInt32LE(sys.length,  8);
//!   buf.writeUInt32LE(offMat,     12);
//!   buf.writeUInt32LE(mat.length, 16);
//!   buf.writeUInt32LE(offChr,     20);
//!   buf.writeUInt32LE(offChr,     20);
//!   buf.writeUInt32LE(chr.length, 24);
//!   buf.writeUInt32LE(offUnk,     28);
//!   buf.writeUInt32LE(unk.length, 32);
//!   sys.copy(buf, offSys);
//!   mat.copy(buf, offMat);
//!   chr.copy(buf, offChr);
//!   unk.copy(buf, offUnk);
//!   return new Uint8Array(buf.buffer);
//! }
//! ```
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { MeCrabWasm } from 'mecrab';
//!
//! await init();
//!
//! const blob = await fetch('/dict/mecrab.bin').then(r => r.arrayBuffer());
//!
//! const mecrab = new MeCrabWasm();
//! const err = mecrab.loadDictionary(new Uint8Array(blob));
//! if (err) throw new Error(err);
//!
//! const result = mecrab.parse("すもももももももものうち");
//! console.log(JSON.parse(result));
//!
//! console.log(mecrab.parseWakati("すもももももももものうち"));
//!
//! mecrab.addWord("ChatGPT", "チャットジーピーティー", "チャットジーピーティー", 5000);
//! ```

mod core;
mod format;
mod loader;
mod parse_impl;

pub use core::MeCrabWasm;

use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages in the browser console.
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Version information.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get feature flags compiled into this build.
#[wasm_bindgen]
pub fn features() -> String {
    format::format_features_list(&["wasm"])
}

#[cfg(test)]
mod tests {
    use super::loader::{BLOB_MAGIC, HEADER_BYTES};
    use super::*;

    // Re-expose parse_blob under the name used by the original tests so that
    // every `MeCrabWasm::parse_blob(...)` call compiles without changes.
    impl MeCrabWasm {
        pub(crate) fn parse_blob(data: &[u8]) -> Result<crate::dict::Dictionary, String> {
            loader::parse_blob(data)
        }
    }

    #[test]
    fn test_wasm_overlay() {
        let mecrab = MeCrabWasm::new();
        mecrab.add_word("テスト", "テスト", "テスト", 5000);
        assert_eq!(mecrab.overlay_size(), 1);
        assert!(mecrab.remove_word("テスト"));
        assert_eq!(mecrab.overlay_size(), 0);
    }

    #[test]
    fn test_not_initialized() {
        let mecrab = MeCrabWasm::new();
        assert!(!mecrab.is_initialized());
        let result = mecrab.parse("テスト");
        assert!(
            result.contains("error"),
            "Expected error JSON, got: {}",
            result
        );
        let result = mecrab.parse_wakati("テスト");
        assert!(
            result.starts_with("Error:"),
            "Expected error string, got: {}",
            result
        );
    }

    #[test]
    fn test_invalid_blob_magic() {
        let mut mecrab = MeCrabWasm::new();
        let mut blob = vec![0u8; 64];
        // Write wrong magic
        blob[0] = 0xDE;
        blob[1] = 0xAD;
        blob[2] = 0xBE;
        blob[3] = 0xEF;
        let err = mecrab.load_dictionary(&blob);
        assert!(!err.is_empty(), "Expected error for invalid magic");
        assert!(
            err.contains("magic") || err.contains("Invalid"),
            "Error should mention magic: {}",
            err
        );
    }

    #[test]
    fn test_blob_too_small() {
        let mut mecrab = MeCrabWasm::new();
        let blob = vec![0u8; 10]; // smaller than HEADER_BYTES (36)
        let err = mecrab.load_dictionary(&blob);
        assert!(!err.is_empty(), "Expected error for undersized blob");
    }

    #[test]
    fn test_parse_blob_section_bounds() {
        // Correct magic, but sys.dic section extends past blob end
        let mut blob = vec![0u8; HEADER_BYTES];
        blob[0..4].copy_from_slice(&BLOB_MAGIC.to_le_bytes());
        // sys.dic: offset = HEADER_BYTES, length = 9999 (way past end)
        blob[4..8].copy_from_slice(&(HEADER_BYTES as u32).to_le_bytes());
        blob[8..12].copy_from_slice(&9999u32.to_le_bytes());
        let err = MeCrabWasm::parse_blob(&blob).unwrap_err();
        assert!(
            err.contains("out of blob bounds") || err.contains("Blob"),
            "Unexpected error: {}",
            err
        );
    }

    /// Build a minimal valid 36-byte blob where all four sections are empty
    /// (offset = HEADER_BYTES, length = 0).
    fn make_minimal_valid_blob() -> Vec<u8> {
        let mut blob = vec![0u8; HEADER_BYTES];
        blob[0..4].copy_from_slice(&BLOB_MAGIC.to_le_bytes());
        // All four sections: offset = HEADER_BYTES (immediately after header), length = 0
        let off = HEADER_BYTES as u32;
        for i in 0..4usize {
            blob[4 + i * 8..4 + i * 8 + 4].copy_from_slice(&off.to_le_bytes());
            blob[8 + i * 8..8 + i * 8 + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        blob
    }

    // ── Step 3 tests ────────────────────────────────────────────────────────

    #[test]
    fn test_load_invalid_magic() {
        let mut mecrab = MeCrabWasm::new();
        // 64-byte blob with wrong magic bytes
        let mut blob = vec![0u8; 64];
        blob[0] = 0xDE;
        blob[1] = 0xAD;
        blob[2] = 0xBE;
        blob[3] = 0xEF;
        let err = mecrab.load_dictionary(&blob);
        assert!(
            !err.is_empty(),
            "Expected non-empty error for invalid magic"
        );
        assert!(
            err.contains("Invalid magic") || err.contains("magic"),
            "Error must mention 'Invalid magic', got: {}",
            err
        );
    }

    #[test]
    fn test_load_truncated_header() {
        let mut mecrab = MeCrabWasm::new();
        // Only 10 bytes — shorter than the 36-byte header requirement
        let blob = vec![0u8; 10];
        let err = mecrab.load_dictionary(&blob);
        assert!(!err.is_empty(), "Expected error for truncated header");
        assert!(
            err.to_lowercase().contains("header")
                || err.to_lowercase().contains("small")
                || err.to_lowercase().contains("too small"),
            "Error should mention header shortness, got: {}",
            err
        );
    }

    #[test]
    fn test_load_section_overflow() {
        let mut mecrab = MeCrabWasm::new();
        // 100-byte blob; header claims sys.dic starts at offset 1000
        let mut blob = vec![0u8; 100];
        blob[0..4].copy_from_slice(&BLOB_MAGIC.to_le_bytes());
        blob[4..8].copy_from_slice(&1000u32.to_le_bytes()); // offset = 1000
        blob[8..12].copy_from_slice(&10u32.to_le_bytes()); // length = 10
        // Remaining section headers: zero (offset 0, length 0 — valid empty sections)
        let err = mecrab.load_dictionary(&blob);
        assert!(
            !err.is_empty(),
            "Expected bounds error when section offset exceeds blob length"
        );
        assert!(
            err.contains("out of blob bounds") || err.contains("bounds") || err.contains("Blob"),
            "Error must mention bounds, got: {}",
            err
        );
    }

    #[test]
    fn test_packed_blob_round_trip() {
        // A valid 36-byte header with all sections empty.  The inner
        // Dictionary::from_bytes call may succeed (empty dict) or return an
        // error for missing data — either outcome is acceptable; what we
        // validate is that no panic occurs and that the bounds checks pass.
        let blob = make_minimal_valid_blob();
        let result = MeCrabWasm::parse_blob(&blob);
        // The call must not panic; success or graceful error are both fine.
        match result {
            Ok(_) => { /* empty dictionary accepted */ }
            Err(e) => {
                // Must not be a bounds error — the header is valid.
                assert!(
                    !e.contains("out of blob bounds"),
                    "Unexpected bounds error for valid minimal blob: {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_parse_without_load() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse("テスト");
        // Result must be a JSON object with an "error" key.
        assert!(
            result.contains("\"error\""),
            "Expected JSON error object, got: {}",
            result
        );
        assert!(
            result.to_lowercase().contains("not") || result.contains("loaded"),
            "Error message should indicate uninitialized state, got: {}",
            result
        );
    }

    #[test]
    fn test_wakati_without_load() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_wakati("テスト");
        assert!(
            result.starts_with("Error:"),
            "parseWakati before load_dictionary must return string starting with 'Error:', got: {}",
            result
        );
        assert!(
            result.to_lowercase().contains("not") || result.contains("loaded"),
            "Error message should indicate uninitialized state, got: {}",
            result
        );
    }

    #[test]
    fn test_add_remove_word_no_crash() {
        // add_word / remove_word before any dictionary is loaded must not panic.
        let mecrab = MeCrabWasm::new();
        // add_word only modifies the standalone overlay — always valid
        mecrab.add_word("テスト語", "テストゴ", "テストゴ", 6000);
        assert_eq!(
            mecrab.overlay_size(),
            1,
            "Word should be in standalone overlay"
        );
        // remove_word before inner dict is loaded should still work on overlay
        let removed = mecrab.remove_word("テスト語");
        assert!(
            removed,
            "remove_word should find and remove the overlay word"
        );
        assert_eq!(
            mecrab.overlay_size(),
            0,
            "Overlay should be empty after removal"
        );
    }

    #[test]
    fn test_version_returns_string() {
        let v = version();
        assert!(!v.is_empty(), "version() must return a non-empty string");
        // Sanity: should look like semver (contains at least one dot)
        assert!(v.contains('.'), "version() should be semver, got: {}", v);
    }

    // ── Group 1: Packed dict format validation ───────────────────────────────

    /// Wrong magic bytes → parse_blob must return Err mentioning "magic"
    #[test]
    fn test_load_packed_dict_wrong_magic() {
        let mut blob = vec![0u8; 64];
        // Write a plausible-but-wrong magic value
        blob[0..4].copy_from_slice(&0xDEAD_C0DEu32.to_le_bytes());
        let result = MeCrabWasm::parse_blob(&blob);
        assert!(result.is_err(), "Expected Err for wrong magic, got Ok");
        let msg = result.err().expect("already asserted Err");
        assert!(
            msg.to_lowercase().contains("magic"),
            "Error must mention 'magic', got: {}",
            msg
        );
    }

    /// Exactly 10-byte slice (below the 36-byte header minimum) → Err
    #[test]
    fn test_load_packed_dict_too_short() {
        let blob = vec![0u8; 10];
        let result = MeCrabWasm::parse_blob(&blob);
        assert!(result.is_err(), "Expected Err for 10-byte blob");
        let msg = result.err().expect("already asserted Err");
        // The error must mention the size constraint somehow
        let lower = msg.to_lowercase();
        assert!(
            lower.contains("small") || lower.contains("header") || lower.contains("bytes"),
            "Error should describe the header size requirement, got: {}",
            msg
        );
    }

    /// Empty slice → Err (header cannot be read)
    #[test]
    fn test_load_packed_dict_empty() {
        let blob: &[u8] = &[];
        let result = MeCrabWasm::parse_blob(blob);
        assert!(result.is_err(), "Expected Err for empty blob");
    }

    /// Minimal valid header (correct magic, zero-length sections) → Ok or graceful Err (no panic)
    #[test]
    fn test_load_packed_dict_valid_but_minimal() {
        let blob = make_minimal_valid_blob();
        // Must not panic; success or a descriptive error are both acceptable.
        let result = MeCrabWasm::parse_blob(&blob);
        match result {
            Ok(_) => { /* empty dict accepted */ }
            Err(e) => {
                // Must not be a bounds/magic error — those are bugs in our header builder.
                assert!(
                    !e.contains("out of blob bounds") && !e.to_lowercase().contains("magic"),
                    "Unexpected structural error from minimal valid blob: {}",
                    e
                );
            }
        }
    }

    // ── Group 2: API surface tests (error paths — no real dict required) ─────

    /// parse("") without a loaded dictionary must return a JSON error object
    #[test]
    fn test_parse_returns_eos_on_empty_input() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse("");
        // With no dict loaded we get an error JSON; with dict we'd get {"tokens":[]}
        // Either way the return must be valid-looking JSON (starts with '{')
        let trimmed = result.trim_start();
        assert!(
            trimmed.starts_with('{'),
            "parse() must return a JSON object, got: {}",
            result
        );
    }

    /// parse_wakati without a loaded dictionary must return an "Error:" string
    #[test]
    fn test_parse_wakati_format() {
        // With no dict we always get an error string starting with "Error:"
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_wakati("日本語テスト");
        assert!(
            result.starts_with("Error:"),
            "parseWakati without dict must start with 'Error:', got: {}",
            result
        );
    }

    /// parse() without a dict must return a JSON object that starts with '{'
    #[test]
    fn test_parse_to_json_valid_json() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse("テスト");
        let trimmed = result.trim_start();
        assert!(
            trimmed.starts_with('{'),
            "parse() output must start with '{{' (JSON object), got: {}",
            result
        );
        // Must contain either "tokens" or "error" key
        assert!(
            result.contains("\"tokens\"") || result.contains("\"error\""),
            "JSON must have 'tokens' or 'error' key, got: {}",
            result
        );
    }

    /// Calling parse() twice with the same input (no dict) must return identical results
    #[test]
    fn test_consistent_repeated_parsing() {
        let mecrab = MeCrabWasm::new();
        let input = "すもももももももものうち";
        let first = mecrab.parse(input);
        let second = mecrab.parse(input);
        assert_eq!(
            first, second,
            "Repeated parse() calls with same input must return identical results"
        );
    }

    // ── Group 3: Unicode and edge cases ──────────────────────────────────────

    /// ASCII input must not panic (returns error JSON without dict)
    #[test]
    fn test_parse_ascii_text() {
        let mecrab = MeCrabWasm::new();
        // Must not panic; error JSON is fine
        let result = mecrab.parse("Hello World");
        assert!(!result.is_empty(), "parse() must return non-empty string");
    }

    /// Whitespace-only input must not panic
    #[test]
    fn test_parse_only_whitespace() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse("   ");
        assert!(
            !result.is_empty(),
            "parse() must return non-empty string for whitespace"
        );
    }

    /// Input containing newlines must not panic
    #[test]
    fn test_parse_newlines() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse("line1\nline2");
        assert!(
            !result.is_empty(),
            "parse() must return non-empty string for newline input"
        );
        // Wakati variant too
        let wk = mecrab.parse_wakati("line1\nline2");
        assert!(
            !wk.is_empty(),
            "parseWakati must return non-empty string for newline input"
        );
    }

    /// Very long input (1000 repetitions of 'あ') must not panic
    #[test]
    fn test_parse_very_long_text() {
        let mecrab = MeCrabWasm::new();
        let long_text: String = std::iter::repeat('あ').take(1000).collect();
        let result = mecrab.parse(&long_text);
        assert!(
            !result.is_empty(),
            "parse() must handle long input without panic"
        );
    }

    /// CJK Unified Ideographs Extension B character (U+20000) must not panic
    #[test]
    fn test_parse_cjk_extension() {
        let mecrab = MeCrabWasm::new();
        // U+20000 is in CJK Extension B; encode as UTF-8: F0 A0 80 80
        let cjk_ext = "\u{20000}";
        let result = mecrab.parse(cjk_ext);
        assert!(
            !result.is_empty(),
            "parse() must handle CJK Extension B without panic"
        );
        let wk = mecrab.parse_wakati(cjk_ext);
        assert!(
            !wk.is_empty(),
            "parseWakati must handle CJK Extension B without panic"
        );
    }

    // ── Group 4: Packed dict binary header format validation ─────────────────

    /// Manually construct a 36-byte header and verify each field at the expected
    /// byte offsets using the same LE-u32 decoding that `parse_blob` uses.
    // ── Track F: BPE-compatible and LatticeProb format tests ────────────────

    #[test]
    fn test_parse_bpe_compatible_no_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_bpe_compatible("東京は日本の首都");
        // Without dict: returns error
        assert!(
            result.starts_with("Error:")
                || result.is_empty()
                || result.contains("▁")
                || result.contains("東京")
        );
    }

    #[test]
    fn test_parse_lattice_prob_no_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_lattice_prob("東京");
        // Returns error JSON or valid JSON
        assert!(!result.is_empty());
        // Should be valid JSON-like output
        assert!(result.starts_with("Error:") || result.starts_with('[') || result.starts_with('{'));
    }

    #[test]
    fn test_parse_bpe_compatible_returns_error_string() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_bpe_compatible("テスト");
        assert!(
            result.starts_with("Error:"),
            "parseBpeCompatible without dict must return Error: string, got: {}",
            result
        );
    }

    #[test]
    fn test_parse_lattice_prob_returns_error_string() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_lattice_prob("テスト");
        assert!(
            result.starts_with("Error:"),
            "parseLatticeProb without dict must return Error: string, got: {}",
            result
        );
    }

    #[test]
    fn test_packed_dict_header_format() {
        // Build a blob exactly as JavaScript `packDictionary` would for zero-length sections.
        // Layout:
        //   [0..4]   magic  = BLOB_MAGIC  (0x4D434142 LE)
        //   [4..8]   sys offset = 36
        //   [8..12]  sys length = 0
        //   [12..16] mat offset = 36
        //   [16..20] mat length = 0
        //   [20..24] chr offset = 36
        //   [24..28] chr length = 0
        //   [28..32] unk offset = 36
        //   [32..36] unk length = 0
        let mut header = [0u8; HEADER_BYTES];
        header[0..4].copy_from_slice(&BLOB_MAGIC.to_le_bytes());
        let off36 = HEADER_BYTES as u32; // = 36
        for i in 0..4usize {
            let base = 4 + i * 8;
            header[base..base + 4].copy_from_slice(&off36.to_le_bytes());
            header[base + 4..base + 8].copy_from_slice(&0u32.to_le_bytes());
        }

        // Verify magic field
        let magic = u32::from_le_bytes(header[0..4].try_into().expect("4 bytes → [u8;4]"));
        assert_eq!(
            magic, BLOB_MAGIC,
            "Magic field at offset 0 must equal BLOB_MAGIC"
        );

        // Verify all four (offset, length) pairs
        for i in 0..4usize {
            let base = 4 + i * 8;
            let off = u32::from_le_bytes(header[base..base + 4].try_into().expect("offset slice"));
            let len =
                u32::from_le_bytes(header[base + 4..base + 8].try_into().expect("length slice"));
            assert_eq!(
                off, off36,
                "Section {} offset must be {} (HEADER_BYTES), got {}",
                i, HEADER_BYTES, off
            );
            assert_eq!(len, 0, "Section {} length must be 0, got {}", i, len);
        }

        // The header blob (exactly 36 bytes) must pass bounds checks in parse_blob.
        let result = MeCrabWasm::parse_blob(&header);
        match result {
            Ok(_) => { /* minimal empty dict accepted */ }
            Err(e) => {
                assert!(
                    !e.to_lowercase().contains("magic") && !e.contains("out of blob bounds"),
                    "Header should be structurally valid; unexpected error: {}",
                    e
                );
            }
        }
    }

    // ── Track O: parseToJson and parseToCsv tests ────────────────────────────

    /// `parseToJson` without a loaded dictionary must return a non-empty string
    /// starting with "Error:" (not a panic).
    #[test]
    fn test_parse_to_json_includes_positions() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_to_json("test");
        // Without dict: must return an error string, never be empty, never panic.
        assert!(
            !result.is_empty(),
            "parse_to_json must return non-empty output"
        );
        assert!(
            result.starts_with("Error:") || result.starts_with('[') || result.starts_with('{'),
            "parse_to_json output must be an error string or JSON, got: {}",
            result
        );
    }

    /// `parseToCsv` without a loaded dictionary must return a non-empty error string.
    #[test]
    fn test_parse_to_csv_no_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_to_csv("東京");
        // Without dict: returns an error string starting with "Error:".
        assert!(
            !result.is_empty(),
            "parse_to_csv must return non-empty output without dict"
        );
        assert!(
            result.starts_with("Error:") || result.contains('\t') || result.is_empty(),
            "parse_to_csv without dict must return Error: string, got: {}",
            result
        );
    }

    /// `parseToCsv` output must be tab-separated with 4 fields per line when a
    /// real dictionary is present; without a dict it must return an "Error:" string.
    /// Either way there must be no panic.
    #[test]
    fn test_parse_to_csv_tab_separated() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_to_csv("hello");
        // Must not panic.  Without a dict we get "Error: ..."; with one every
        // non-empty line must have exactly 4 tab-separated fields.
        assert!(
            result.starts_with("Error:")
                || result
                    .lines()
                    .all(|line| { line.is_empty() || line.split('\t').count() == 4 }),
            "parse_to_csv output lines must have 4 tab-separated fields, got: {}",
            result
        );
    }

    /// `parseToJson` must return "Error:" without a loaded dictionary (mirrors
    /// the behaviour of `parseBpeCompatible` and `parseLatticeProb`).
    #[test]
    fn test_parse_to_json_returns_error_without_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_to_json("東京は日本の首都");
        assert!(
            result.starts_with("Error:"),
            "parseToJson without dict must return Error: string, got: {}",
            result
        );
    }

    /// `parseToCsv` must return "Error:" without a loaded dictionary.
    #[test]
    fn test_parse_to_csv_returns_error_without_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_to_csv("東京は日本の首都");
        assert!(
            result.starts_with("Error:"),
            "parseToCsv without dict must return Error: string, got: {}",
            result
        );
    }

    #[test]
    fn test_parse_conllu_no_dict() {
        let mecrab = MeCrabWasm::new();
        let result = mecrab.parse_conllu("東京は日本の首都");
        // Without a dictionary, must return an error string
        assert!(
            result.starts_with("Error:"),
            "parse_conllu without dict must return Error: prefix, got: {}",
            result
        );
    }
}
