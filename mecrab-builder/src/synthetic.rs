//! Synthetic in-memory dictionary for tests, benchmarks, and WASM.
//!
//! `build_synthetic_dictionary()` constructs a minimal but correct MeCab-format
//! dictionary entirely in RAM, with no filesystem access required.  It provides
//! a curated Japanese lexicon that enables end-to-end pipeline testing without
//! an IPADIC installation.
//!
//! # Curated lexicon
//!
//! The dictionary covers the classic segmentation sentence
//! 「すもももももももものうち」 and a few additional entries for testing.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::char_writer::{CharRange, build_char_bytes, pack_char_info};
use crate::matrix_writer::{build_matrix_bytes, set_cost};
use crate::sysdic_writer::{DicEntry, build_sysdic_bytes, build_unkdic_bytes};

/// The four binary byte buffers that constitute a complete MeCab-format dictionary.
pub struct SyntheticDictionary {
    /// sys.dic bytes (dict_type=0)
    pub sys_dic: Vec<u8>,
    /// matrix.bin bytes
    pub matrix: Vec<u8>,
    /// char.bin bytes
    pub char_def: Vec<u8>,
    /// unk.dic bytes (dict_type=2)
    pub unk_def: Vec<u8>,
}

impl SyntheticDictionary {
    /// Parse the four byte buffers into a `mecrab::dict::Dictionary`.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer is corrupted or has an invalid format.
    pub fn load(&self) -> mecrab::Result<mecrab::dict::Dictionary> {
        mecrab::dict::Dictionary::from_bytes(
            &self.sys_dic,
            &self.matrix,
            &self.char_def,
            &self.unk_def,
        )
    }

    /// Construct a `MeCrab` analyzer from the four byte buffers.
    ///
    /// # Errors
    ///
    /// Returns an error if any buffer is corrupted or has an invalid format.
    pub fn into_mecrab(self) -> mecrab::Result<mecrab::MeCrab> {
        mecrab::MeCrab::from_bytes(&self.sys_dic, &self.matrix, &self.char_def, &self.unk_def)
    }
}

// ── Context-ID scheme ────────────────────────────────────────────
// 0 = BOS / EOS (reserved)
// 1 = 名詞 (noun)
// 2 = 助詞 (particle)
//
// lsize = rsize = 8  (leaves 3–7 for future expansion)

const LSIZE: u16 = 8;
const RSIZE: u16 = 8;

// ── sys.dic entries ──────────────────────────────────────────────

fn sys_entries() -> Vec<DicEntry> {
    vec![
        DicEntry {
            surface: "すもも".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 5000,
            feature: "名詞,一般,*,*,*,*,すもも,スモモ,スモモ".to_string(),
        },
        DicEntry {
            surface: "もも".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 5000,
            feature: "名詞,一般,*,*,*,*,もも,モモ,モモ".to_string(),
        },
        DicEntry {
            surface: "うち".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 5000,
            feature: "名詞,非自立,副詞可能,*,*,*,うち,ウチ,ウチ".to_string(),
        },
        DicEntry {
            surface: "東京".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 4500,
            feature: "名詞,固有名詞,地名,一般,*,*,東京,トウキョウ,トウキョウ".to_string(),
        },
        DicEntry {
            surface: "日本".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 4500,
            feature: "名詞,固有名詞,地名,一般,*,*,日本,ニホン,ニホン".to_string(),
        },
        DicEntry {
            surface: "人".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 5500,
            feature: "名詞,接尾,一般,*,*,*,人,ニン,ニン".to_string(),
        },
        DicEntry {
            surface: "も".to_string(),
            left_id: 2,
            right_id: 2,
            pos_id: 2,
            wcost: 4000,
            feature: "助詞,係助詞,*,*,*,*,も,モ,モ".to_string(),
        },
        DicEntry {
            surface: "の".to_string(),
            left_id: 2,
            right_id: 2,
            pos_id: 2,
            wcost: 3000,
            feature: "助詞,連体化,*,*,*,*,の,ノ,ノ".to_string(),
        },
        DicEntry {
            surface: "は".to_string(),
            left_id: 2,
            right_id: 2,
            pos_id: 2,
            wcost: 3500,
            feature: "助詞,係助詞,*,*,*,*,は,ハ,ワ".to_string(),
        },
    ]
}

// ── Connection matrix ────────────────────────────────────────────
//
// 8×8 = 64 entries.  Default cost = 1000.
// Customised transitions:
//   cost(right=0, left=1) = 0     BOS → noun
//   cost(right=0, left=2) = 200   BOS → particle
//   cost(right=1, left=0) = 0     noun → EOS
//   cost(right=2, left=0) = 400   particle → EOS
//   cost(right=1, left=2) = 0     noun → particle (very cheap)
//   cost(right=2, left=1) = 0     particle → noun (very cheap)
//   cost(right=1, left=1) = 1500  noun → noun (expensive, breaks greedy)
//   cost(right=2, left=2) = 1500  particle → particle (expensive)
//
// The ViterbiSolver looks up cost(right_id_of_left_node, left_id_of_right_node).
// Index = right_id + lsize * left_id

fn matrix_costs() -> Vec<i16> {
    let lsize = LSIZE as usize;
    let n = lsize * RSIZE as usize;
    let mut costs = vec![1000i16; n];

    set_cost(&mut costs, lsize, 0, 1, 0); // BOS → noun
    set_cost(&mut costs, lsize, 0, 2, 200); // BOS → particle
    set_cost(&mut costs, lsize, 1, 0, 0); // noun → EOS
    set_cost(&mut costs, lsize, 2, 0, 400); // particle → EOS
    set_cost(&mut costs, lsize, 1, 2, 0); // noun → particle
    set_cost(&mut costs, lsize, 2, 1, 0); // particle → noun
    set_cost(&mut costs, lsize, 1, 1, 1500); // noun → noun (expensive)
    set_cost(&mut costs, lsize, 2, 2, 1500); // particle → particle (expensive)

    costs
}

// ── char.bin ─────────────────────────────────────────────────────
//
// Categories in CharCategory enum-id order:
//   0=DEFAULT, 1=SPACE, 2=KANJI, 3=SYMBOL, 4=NUMERIC, 5=ALPHA,
//   6=HIRAGANA, 7=KATAKANA, 8=KANJINUMERIC, 9=GREEK, 10=CYRILLIC
//
// For each category id, the type_mask = (1 << id), default_type = id,
// group=true (so should_group() works), invoke=true.
//
// Representative characters for should_group() (char_def.rs:271-288):
//   DEFAULT/SPACE: ' ' U+0020
//   KANJI: '漢' U+6F22  → inside U+4E00..=U+9FFF ✓
//   SYMBOL: '!' U+0021  → inside U+3000..=U+303F? No — we add U+0021 to SYMBOL too
//   NUMERIC: '0' U+0030 → inside U+0030..=U+0039 ✓
//   ALPHA: 'A' U+0041   → inside U+0041..=U+005A ✓
//   HIRAGANA: 'あ' U+3042 → inside U+3041..=U+3096 ✓
//   KATAKANA: 'ア' U+30A2 → inside U+30A1..=U+30FC ✓
//   KANJINUMERIC: '一' U+4E00 → inside KANJI range (needs own mapping)
//   GREEK: 'Α' U+0391    → need to add range
//   CYRILLIC: 'А' U+0410 → need to add range

const CATEGORY_NAMES: &[&str] = &[
    "DEFAULT",      // 0
    "SPACE",        // 1
    "KANJI",        // 2
    "SYMBOL",       // 3
    "NUMERIC",      // 4
    "ALPHA",        // 5
    "HIRAGANA",     // 6
    "KATAKANA",     // 7
    "KANJINUMERIC", // 8
    "GREEK",        // 9
    "CYRILLIC",     // 10
];

fn make_char_info(id: usize) -> u32 {
    pack_char_info(
        1u32 << id, // type_mask: exactly bit `id`
        id as u8,   // default_type: category id
        0,          // length: 0 (no explicit limit)
        true,       // group: true (group consecutive same-category chars)
        true,       // invoke: true (invoke unknown-word processing)
    )
}

fn char_ranges() -> Vec<CharRange> {
    vec![
        // id=1 (SPACE): U+0020 (space)
        CharRange {
            lo: 0x0020,
            hi: 0x0020,
            info: make_char_info(1),
        },
        // id=3 (SYMBOL): U+0021 ('!') — needed for should_group representative
        //   Also U+3000..=U+303F (CJK symbols and punctuation)
        CharRange {
            lo: 0x0021,
            hi: 0x0021,
            info: make_char_info(3),
        },
        CharRange {
            lo: 0x3000,
            hi: 0x303F,
            info: make_char_info(3),
        },
        // id=4 (NUMERIC): U+0030..=U+0039 ('0'..'9')
        CharRange {
            lo: 0x0030,
            hi: 0x0039,
            info: make_char_info(4),
        },
        // id=5 (ALPHA): U+0041..=U+005A ('A'..'Z'), U+0061..=U+007A ('a'..'z')
        CharRange {
            lo: 0x0041,
            hi: 0x005A,
            info: make_char_info(5),
        },
        CharRange {
            lo: 0x0061,
            hi: 0x007A,
            info: make_char_info(5),
        },
        // id=6 (HIRAGANA): U+3041..=U+3096
        // Covers 'あ' (U+3042), 'す'(U+3059), 'も'(U+3082), 'の'(U+306E),
        //         'う'(U+3046), 'ち'(U+3061), 'は'(U+306F)
        CharRange {
            lo: 0x3041,
            hi: 0x3096,
            info: make_char_info(6),
        },
        // id=7 (KATAKANA): U+30A1..=U+30FC
        // Covers 'ア'(U+30A2), 'グ'(U+30B0), 'ー'(U+30FC) etc.
        CharRange {
            lo: 0x30A1,
            hi: 0x30FC,
            info: make_char_info(7),
        },
        // id=8 (KANJINUMERIC): U+4E00 ('一') alone — representative char
        // We assign it KANJINUMERIC to disambiguate from KANJI.
        // Note: '一' = U+4E00 is the first CJK ideograph.
        CharRange {
            lo: 0x4E00,
            hi: 0x4E00,
            info: make_char_info(8),
        },
        // id=2 (KANJI): U+4E01..=U+9FFF (rest of CJK block, after '一')
        // Covers '漢'(U+6F22), '東'(U+6771), '京'(U+4EAC), '日'(U+65E5), '本'(U+672C),
        //        '人'(U+4EBA)
        CharRange {
            lo: 0x4E01,
            hi: 0x9FFF,
            info: make_char_info(2),
        },
        // id=9 (GREEK): U+0391..=U+03C9 (Greek and Coptic block)
        // Covers 'Α'(U+0391)
        CharRange {
            lo: 0x0391,
            hi: 0x03C9,
            info: make_char_info(9),
        },
        // id=10 (CYRILLIC): U+0410..=U+044F (Cyrillic block)
        // Covers 'А'(U+0410)
        CharRange {
            lo: 0x0410,
            hi: 0x044F,
            info: make_char_info(10),
        },
    ]
}

// ── unk.dic entries ──────────────────────────────────────────────
//
// Keys are uppercase category names as in `unknown.rs:113-125`.
// dict_type must be 2 (MECAB_UNK_DIC).

fn unk_entries() -> Vec<DicEntry> {
    vec![
        DicEntry {
            surface: "DEFAULT".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 10000,
            feature: "記号,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "SPACE".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 8000,
            feature: "記号,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "KANJI".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 6000,
            feature: "名詞,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "SYMBOL".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 9000,
            feature: "記号,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "NUMERIC".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 7000,
            feature: "名詞,数,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "ALPHA".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 7000,
            feature: "名詞,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "HIRAGANA".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 8000,
            feature: "名詞,一般,*,*,*,*,*,*,*".to_string(),
        },
        DicEntry {
            surface: "KATAKANA".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 1,
            wcost: 7000,
            feature: "名詞,一般,*,*,*,*,*,*,*".to_string(),
        },
    ]
}

// ── Public constructor ───────────────────────────────────────────

/// Build a minimal but fully correct synthetic MeCab-format dictionary entirely
/// in RAM.
///
/// The resulting [`SyntheticDictionary`] can be passed directly to
/// `mecrab::MeCrab::from_bytes` for end-to-end testing without any disk
/// dictionary.
///
/// # Panics
///
/// Panics if any of the internal serialization steps fail (should not happen
/// with the hard-coded data).
#[must_use]
pub fn build_synthetic_dictionary() -> SyntheticDictionary {
    // sys.dic (dict_type = 0)
    let entries = sys_entries();
    let (sys_dic, _) = build_sysdic_bytes(&entries, LSIZE as u32, RSIZE as u32, "UTF-8", 0)
        .expect("build sys.dic failed");

    // matrix.bin
    let costs = matrix_costs();
    let matrix = build_matrix_bytes(LSIZE, RSIZE, &costs).expect("build matrix.bin failed");

    // char.bin
    let char_def = build_char_bytes(CATEGORY_NAMES, &char_ranges()).expect("build char.bin failed");

    // unk.dic (dict_type = 2)
    let unk_entries_data = unk_entries();
    let (unk_def, _) = build_unkdic_bytes(&unk_entries_data, LSIZE as u32, RSIZE as u32, "UTF-8")
        .expect("build unk.dic failed");

    SyntheticDictionary {
        sys_dic,
        matrix,
        char_def,
        unk_def,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_synthetic_dictionary_no_panic() {
        let d = build_synthetic_dictionary();
        assert!(!d.sys_dic.is_empty());
        assert!(!d.matrix.is_empty());
        assert!(!d.char_def.is_empty());
        assert!(!d.unk_def.is_empty());
    }

    #[test]
    fn test_unk_dic_type_field() {
        let d = build_synthetic_dictionary();
        // dict_type is at bytes [8..12]
        let dict_type =
            u32::from_le_bytes([d.unk_def[8], d.unk_def[9], d.unk_def[10], d.unk_def[11]]);
        assert_eq!(dict_type, 2, "unk.dic dict_type must be 2 (MECAB_UNK_DIC)");
    }

    #[test]
    fn test_sys_dic_type_field() {
        let d = build_synthetic_dictionary();
        let dict_type =
            u32::from_le_bytes([d.sys_dic[8], d.sys_dic[9], d.sys_dic[10], d.sys_dic[11]]);
        assert_eq!(dict_type, 0, "sys.dic dict_type must be 0 (MECAB_SYS_DIC)");
    }

    #[test]
    fn test_matrix_size() {
        let d = build_synthetic_dictionary();
        // Expected: 4 + 8*8*2 = 132 bytes
        assert_eq!(d.matrix.len(), 4 + 8 * 8 * 2);
    }

    #[test]
    fn test_char_bin_size() {
        let d = build_synthetic_dictionary();
        let csize = CATEGORY_NAMES.len();
        let expected = 4 + csize * 32 + 0xFFFF * 4;
        assert_eq!(
            d.char_def.len(),
            expected,
            "char.bin must be exactly {expected} bytes"
        );
    }

    #[test]
    fn test_load_dictionary() {
        let d = build_synthetic_dictionary();
        let dict = d.load();
        assert!(
            dict.is_ok(),
            "dictionary load must succeed: {:?}",
            dict.err()
        );
    }

    #[test]
    fn test_into_mecrab() {
        let d = build_synthetic_dictionary();
        let m = d.into_mecrab();
        assert!(m.is_ok(), "MeCrab::from_bytes must succeed: {:?}", m.err());
    }
}
