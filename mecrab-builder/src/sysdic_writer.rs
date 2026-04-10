//! sys.dic binary format writer for MeCab-compatible dictionaries.
//!
//! Generates a MeCab-compatible sys.dic from a list of dictionary entries,
//! producing a binary file that can be loaded by the SysDic reader in
//! the mecrab core crate.
//!
//! # Binary Format
//! See sys_dic.rs in the mecrab crate for the complete format specification.
//!
//! # Double-Array Construction
//! Uses Aoe's (1989) algorithm for constructing a Double-Array Trie from
//! a set of sorted keys. The construction runs in O(N * max_key_len) where
//! N is the number of distinct keys.
//!
//! # DA Semantics (must match double_array_trie.rs reader)
//!
//! The reader uses this invariant:
//!   - `b = units[0].base`  (root base)
//!   - Transition via byte `c`: slot `p = b + c + 1`
//!     valid if `units[p].check == b as u32`, then `b = units[p].base`
//!   - Terminal: at `p = b as usize`, if `units[p].check == b as u32`
//!     and `units[p].base < 0` → value = `-units[p].base - 1`
//!
//! So a "value node" is a self-referential slot where check == its own index.

// Copyright 2026 COOLJAPAN OU (Team KitaSan)

use crate::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use std::collections::HashMap;
use std::path::Path;

// ── Constants matching sys_dic.rs ───────────────────────────────

const DICTIONARY_MAGIC_ID: u32 = 0xef71_8f77;
const DIC_VERSION: u32 = 102;
const HEADER_SIZE: usize = 72;

// Sentinel meaning "this slot is not yet allocated"
const FREE_CHECK: u32 = u32::MAX;

// ── Public input type ────────────────────────────────────────────

/// A single dictionary entry to include in the sys.dic
#[derive(Debug, Clone)]
pub struct DicEntry {
    /// Surface form (the word as it appears in text)
    pub surface: String,
    /// Left context attribute
    pub left_id: u16,
    /// Right context attribute
    pub right_id: u16,
    /// Part-of-speech ID (0 if not applicable)
    pub pos_id: u16,
    /// Word cost
    pub wcost: i16,
    /// Feature string (e.g. "名詞,固有名詞,地名,一般,*,*,東京,トウキョウ,トウキョウ")
    pub feature: String,
}

// ── Statistics ───────────────────────────────────────────────────

/// Statistics from a sys.dic write operation
#[derive(Debug, Default)]
pub struct WriteSysDicStats {
    pub lexicon_size: u32,
    pub token_count: usize,
    pub da_units: usize,
    pub feature_bytes: usize,
    pub file_size: usize,
}

// ── Internal token bytes ─────────────────────────────────────────

struct TokenBytes {
    left_id: u16,
    right_id: u16,
    pos_id: u16,
    wcost: i16,
    feature_offset: u32,
    compound: u32,
}

// ── Trie node ────────────────────────────────────────────────────

/// A node in the intermediate trie used before DA construction.
#[derive(Default)]
struct TrieNode {
    /// Children indexed by byte value (0..=255)
    children: HashMap<u8, usize>,
    /// Stored DA value at this terminal node; i32::MIN = not terminal
    value: i32,
}

impl TrieNode {
    fn is_terminal(&self) -> bool {
        self.value != i32::MIN
    }
}

/// Build an intermediate trie from (key_bytes, value) pairs.
struct TrieBuilder {
    nodes: Vec<TrieNode>,
}

impl TrieBuilder {
    fn new() -> Self {
        Self {
            nodes: vec![TrieNode {
                value: i32::MIN,
                ..Default::default()
            }],
        }
    }

    fn insert(&mut self, key: &[u8], value: i32) {
        let mut state = 0usize;
        for &byte in key {
            let child = if let Some(&c) = self.nodes[state].children.get(&byte) {
                c
            } else {
                let new_id = self.nodes.len();
                self.nodes.push(TrieNode {
                    value: i32::MIN,
                    ..Default::default()
                });
                self.nodes[state].children.insert(byte, new_id);
                new_id
            };
            state = child;
        }
        self.nodes[state].value = value;
    }
}

// ── DA builder ───────────────────────────────────────────────────

/// Double-Array unit — 8 bytes, matches the `Unit` struct in the reader.
#[derive(Clone, Copy, Default)]
struct DaUnit {
    base: i32,
    check: u32,
}

/// Builds a Double-Array whose semantics exactly match the reader in
/// `double_array_trie.rs`:
///
/// ```text
/// transition(state_b, byte_c) → slot  p = b + c + 1
///            valid iff  units[p].check == b as u32
///            new b     = units[p].base
///
/// terminal at b  →  units[b].check == b as u32 && units[b].base < 0
///                   value = -units[b].base - 1
/// ```
///
/// We use BFS so that parents are fully placed before children.
struct DaBuilder {
    units: Vec<DaUnit>,
}

impl DaBuilder {
    fn new() -> Self {
        // Seed with enough room; we'll grow as needed.
        // Index 0 is the root; unit[0].base = the root's BASE value.
        // unit[0].check is never read by the reader (it starts from units[0].base).
        let units = vec![
            DaUnit {
                base: 0,
                check: FREE_CHECK,
            };
            65536
        ];
        Self { units }
    }

    fn ensure_capacity(&mut self, idx: usize) {
        if idx >= self.units.len() {
            let new_size = ((idx + 1).next_power_of_two()).max(self.units.len() * 2);
            self.units.resize(
                new_size,
                DaUnit {
                    base: 0,
                    check: FREE_CHECK,
                },
            );
        }
    }

    #[inline]
    fn is_free(&self, idx: usize) -> bool {
        idx >= self.units.len() || self.units[idx].check == FREE_CHECK
    }

    /// Find a BASE value `b` such that:
    ///   - For every child byte `c` in `child_bytes`:  slot `b + c + 1` is free
    ///   - For terminal:                               slot `b`          is free
    ///
    /// `need_terminal` is true when this node is also a terminal (needs a
    /// self-referential value slot at `b`).
    fn find_base(&self, child_bytes: &[u8], need_terminal: bool) -> usize {
        // We want the smallest b >= 1 such that all required slots are free.
        // Required child slots: b + c + 1  for each c in child_bytes
        // Required terminal slot (if need_terminal): b  itself
        let mut b: usize = 1;
        'outer: loop {
            // Terminal self-reference: units[b].check must be free
            if need_terminal && !self.is_free(b) {
                b += 1;
                continue 'outer;
            }
            for &c in child_bytes {
                let slot = b + c as usize + 1;
                if !self.is_free(slot) {
                    b += 1;
                    continue 'outer;
                }
            }
            return b;
        }
    }

    /// Recursively build the DA from a TrieBuilder using BFS.
    ///
    /// `da_state` is the index in the DA array whose `.base` we are computing.
    /// The reader traverses:  `b = units[da_state].base`  and then looks up children.
    ///
    /// So `da_state` is the "owning slot" of the node; its `.base` is what we set here.
    fn build_from_trie(&mut self, trie: &TrieBuilder) {
        // Queue entries: (trie_node_id, da_slot_index)
        // da_slot_index is the slot whose `.base` we'll set to BASE of this node.
        let mut queue: std::collections::VecDeque<(usize, usize)> =
            std::collections::VecDeque::new();

        // The root trie node lives at da_slot 0.
        // We need to set units[0].base = BASE_root.
        queue.push_back((0usize, 0usize));

        while let Some((trie_node, da_slot)) = queue.pop_front() {
            let node = &trie.nodes[trie_node];

            let mut child_bytes: Vec<u8> = node.children.keys().copied().collect();
            child_bytes.sort_unstable();

            let need_terminal = node.is_terminal();

            if child_bytes.is_empty() && !need_terminal {
                // Pure leaf that is terminal: handled via need_terminal from parent.
                // (Shouldn't normally be reached because we always set need_terminal.)
                continue;
            }

            let base = self.find_base(&child_bytes, need_terminal);

            // Write this node's BASE into its DA slot
            self.ensure_capacity(da_slot);
            self.units[da_slot].base = base as i32;

            // If terminal: place a self-referential value slot at `base`.
            // units[base].check = base  (self-reference)
            // units[base].base  = -value - 1  (leaf encoding)
            if need_terminal {
                self.ensure_capacity(base);
                // Mark slot `base` as claimed by the value-slot convention.
                // check = base  (so reader: units[base].check == b as u32  where b == base)
                self.units[base].check = base as u32;
                self.units[base].base = -node.value - 1;
            }

            // Place child transition slots: units[base + c + 1]
            for &c in &child_bytes {
                let slot = base + c as usize + 1;
                self.ensure_capacity(slot);
                // check = base  (parent's base)
                // base  = TBD — will be filled when we process that child
                self.units[slot].check = base as u32;
                // Default base; BFS will overwrite when this child is processed
                self.units[slot].base = 0;

                let child_trie_node = trie.nodes[trie_node].children[&c];
                queue.push_back((child_trie_node, slot));
            }
        }
    }

    fn finish(mut self) -> Vec<DaUnit> {
        // Trim trailing free slots
        while self.units.last().is_some_and(|u| u.check == FREE_CHECK) {
            self.units.pop();
        }
        // Replace remaining FREE_CHECK sentinels with u32::MAX for the file
        // (invalid check values that will never match)
        for unit in &mut self.units {
            if unit.check == FREE_CHECK {
                unit.check = u32::MAX;
            }
        }
        self.units
    }
}

// ── Main writer ──────────────────────────────────────────────────

/// Write a MeCab-compatible sys.dic binary file.
///
/// # Arguments
/// * `entries`     - Dictionary entries (sorted by surface internally)
/// * `left_size`   - Number of left context IDs (from connection matrix)
/// * `right_size`  - Number of right context IDs
/// * `charset`     - Character set string (e.g., `"UTF-8"`)
/// * `output`      - Output file path
///
/// # Errors
///
/// Returns [`BuildError`] on I/O or serialization failure.
pub fn write_sysdic(
    entries: &[DicEntry],
    left_size: u32,
    right_size: u32,
    charset: &str,
    output: &Path,
) -> Result<WriteSysDicStats> {
    // ── Group entries by surface (BTreeMap → sorted order) ──────
    let mut by_surface: std::collections::BTreeMap<Vec<u8>, Vec<&DicEntry>> =
        std::collections::BTreeMap::new();
    for entry in entries {
        by_surface
            .entry(entry.surface.as_bytes().to_vec())
            .or_default()
            .push(entry);
    }

    // ── Build token array + feature section ─────────────────────
    let mut tokens: Vec<TokenBytes> = Vec::with_capacity(entries.len());
    let mut feature_bytes: Vec<u8> = Vec::new();
    let mut feature_offsets: HashMap<String, u32> = HashMap::new();

    // (surface_key → da_value) pairs for trie construction
    let mut da_entries: Vec<(Vec<u8>, i32)> = Vec::with_capacity(by_surface.len());
    let mut lexicon_size: u32 = 0;

    for (surface_bytes, surface_entries) in &by_surface {
        let count = surface_entries.len().min(255);
        let token_start = tokens.len();

        // DA value encoding: upper bits = token_start, lower 8 bits = count
        let da_value = ((token_start as i32) << 8) | (count as i32);
        da_entries.push((surface_bytes.clone(), da_value));
        lexicon_size += 1;

        for &entry in surface_entries.iter().take(255) {
            let feature_offset = if let Some(&off) = feature_offsets.get(&entry.feature) {
                off
            } else {
                let off = feature_bytes.len() as u32;
                feature_offsets.insert(entry.feature.clone(), off);
                feature_bytes.extend_from_slice(entry.feature.as_bytes());
                feature_bytes.push(0); // NUL terminator
                off
            };

            tokens.push(TokenBytes {
                left_id: entry.left_id,
                right_id: entry.right_id,
                pos_id: entry.pos_id,
                wcost: entry.wcost,
                feature_offset,
                compound: 0,
            });
        }
    }

    // ── Build Double-Array ───────────────────────────────────────
    let mut trie = TrieBuilder::new();
    for (key, value) in &da_entries {
        trie.insert(key, *value);
    }

    let mut da_builder = DaBuilder::new();
    da_builder.build_from_trie(&trie);
    let da_units = da_builder.finish();

    // ── Compute section sizes ────────────────────────────────────
    let da_size = da_units.len() * 8; // 8 bytes per unit
    let token_size = tokens.len() * 16; // 16 bytes per token
    let feature_size = feature_bytes.len();

    let total_size = HEADER_SIZE + da_size + token_size + feature_size;
    let magic = (total_size as u32) ^ DICTIONARY_MAGIC_ID;

    // ── Serialize ────────────────────────────────────────────────
    let mut buf: Vec<u8> = Vec::with_capacity(total_size);

    // Header (10 × u32 = 40 bytes)
    buf.write_u32::<LittleEndian>(magic)?;
    buf.write_u32::<LittleEndian>(DIC_VERSION)?;
    buf.write_u32::<LittleEndian>(0)?; // dict_type = sys
    buf.write_u32::<LittleEndian>(lexicon_size)?;
    buf.write_u32::<LittleEndian>(left_size)?;
    buf.write_u32::<LittleEndian>(right_size)?;
    buf.write_u32::<LittleEndian>(da_size as u32)?;
    buf.write_u32::<LittleEndian>(token_size as u32)?;
    buf.write_u32::<LittleEndian>(feature_size as u32)?;
    buf.write_u32::<LittleEndian>(0)?; // dummy/padding

    // Charset (32 bytes, null-padded)
    let mut charset_field = [0u8; 32];
    let cs = charset.as_bytes();
    let cs_len = cs.len().min(31);
    charset_field[..cs_len].copy_from_slice(&cs[..cs_len]);
    buf.extend_from_slice(&charset_field);

    // DA section
    for unit in &da_units {
        buf.write_i32::<LittleEndian>(unit.base)?;
        buf.write_u32::<LittleEndian>(unit.check)?;
    }

    // Token section (16 bytes each)
    for token in &tokens {
        buf.write_u16::<LittleEndian>(token.left_id)?;
        buf.write_u16::<LittleEndian>(token.right_id)?;
        buf.write_u16::<LittleEndian>(token.pos_id)?;
        buf.write_i16::<LittleEndian>(token.wcost)?;
        buf.write_u32::<LittleEndian>(token.feature_offset)?;
        buf.write_u32::<LittleEndian>(token.compound)?;
    }

    // Feature section
    buf.extend_from_slice(&feature_bytes);

    // Write to file
    std::fs::write(output, &buf)?;

    Ok(WriteSysDicStats {
        lexicon_size,
        token_count: tokens.len(),
        da_units: da_units.len(),
        feature_bytes: feature_size,
        file_size: total_size,
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entries() -> Vec<DicEntry> {
        vec![
            DicEntry {
                surface: "東京".to_string(),
                left_id: 1285,
                right_id: 1285,
                pos_id: 38,
                wcost: 3780,
                feature: "名詞,固有名詞,地名,一般,*,*,東京,トウキョウ,トウキョウ".to_string(),
            },
            DicEntry {
                surface: "大阪".to_string(),
                left_id: 1285,
                right_id: 1285,
                pos_id: 38,
                wcost: 4900,
                feature: "名詞,固有名詞,地名,一般,*,*,大阪,オオサカ,オオサカ".to_string(),
            },
            DicEntry {
                surface: "東京都".to_string(),
                left_id: 1285,
                right_id: 1285,
                pos_id: 38,
                wcost: 3200,
                feature: "名詞,固有名詞,地名,一般,*,*,東京都,トウキョウト,トウキョウト".to_string(),
            },
            DicEntry {
                surface: "東".to_string(),
                left_id: 1285,
                right_id: 1285,
                pos_id: 38,
                wcost: 6000,
                feature: "名詞,固有名詞,地名,一般,*,*,東,ヒガシ,ヒガシ".to_string(),
            },
        ]
    }

    #[test]
    fn test_write_sysdic_produces_valid_file() {
        let entries = make_entries();
        let dir = std::env::temp_dir();
        let output = dir.join("test_sys.dic");

        let stats = write_sysdic(&entries, 1316, 1316, "UTF-8", &output)
            .expect("write_sysdic should succeed");

        assert_eq!(stats.lexicon_size, 4, "lexicon_size should be 4");
        assert_eq!(stats.token_count, 4, "token_count should be 4");
        assert!(
            stats.file_size >= HEADER_SIZE,
            "file_size should be at least header size"
        );
        assert!(stats.da_units > 0, "da_units should be positive");

        // Verify magic number in file
        let data = std::fs::read(&output).expect("should read output file");
        let magic = u32::from_le_bytes(data[0..4].try_into().expect("4 bytes"));
        let expected_size = (magic ^ DICTIONARY_MAGIC_ID) as usize;
        assert_eq!(
            expected_size,
            data.len(),
            "Magic number mismatch: file size = {}, magic implies {}",
            data.len(),
            expected_size
        );

        // Verify version
        let version = u32::from_le_bytes(data[4..8].try_into().expect("4 bytes"));
        assert_eq!(version, DIC_VERSION, "version should match DIC_VERSION");

        std::fs::remove_file(&output).ok();
    }

    #[test]
    fn test_trie_builder_insert() {
        let mut trie = TrieBuilder::new();
        trie.insert(b"hello", 42);
        trie.insert(b"hell", 10);
        trie.insert(b"world", 99);

        assert!(
            trie.nodes[0].children.contains_key(&b'h'),
            "'h' child should exist at root"
        );
        let h_node = trie.nodes[0].children[&b'h'];
        assert!(
            trie.nodes[h_node].children.contains_key(&b'e'),
            "'e' child should exist"
        );
    }

    #[test]
    fn test_da_value_encoding() {
        let token_start: usize = 42;
        let count: usize = 3;
        let da_value: i32 = ((token_start << 8) | count) as i32;

        let decoded_start = (da_value as u32 >> 8) as usize;
        let decoded_count = (da_value as u32 & 0xff) as usize;

        assert_eq!(decoded_start, token_start);
        assert_eq!(decoded_count, count);
    }

    #[test]
    fn test_write_sysdic_roundtrip_magic() {
        let entries = vec![DicEntry {
            surface: "a".to_string(),
            left_id: 1,
            right_id: 1,
            pos_id: 0,
            wcost: 100,
            feature: "test".to_string(),
        }];

        let dir = std::env::temp_dir();
        let output = dir.join("test_mini.dic");
        let stats =
            write_sysdic(&entries, 10, 10, "UTF-8", &output).expect("write_sysdic should succeed");

        let data = std::fs::read(&output).expect("should read output file");
        assert_eq!(data.len(), stats.file_size, "file size should match stats");

        let magic = u32::from_le_bytes(data[0..4].try_into().expect("4 bytes"));
        assert_eq!(
            (magic ^ DICTIONARY_MAGIC_ID) as usize,
            data.len(),
            "magic round-trip should hold"
        );

        std::fs::remove_file(&output).ok();
    }

    /// Verify DA structure for a minimal trie manually.
    ///
    /// Insert "ab" → 999. Then:
    ///   b₀ = units[0].base
    ///   slot_a = b₀ + 'a'(97) + 1  → units[slot_a].check == b₀
    ///   b₁ = units[slot_a].base
    ///   slot_b = b₁ + 'b'(98) + 1  → units[slot_b].check == b₁
    ///   b₂ = units[slot_b].base
    ///   terminal: units[b₂].check == b₂  && units[b₂].base == -999 - 1 = -1000
    #[test]
    fn test_da_traversal_invariant() {
        let mut trie = TrieBuilder::new();
        trie.insert(b"ab", 999);

        let mut da_builder = DaBuilder::new();
        da_builder.build_from_trie(&trie);
        let units = da_builder.finish();

        // Root: b₀ = units[0].base
        let b0 = units[0].base as usize;

        // Transition via 'a' (97): slot = b0 + 97 + 1
        let slot_a = b0 + 97 + 1;
        assert!(
            slot_a < units.len(),
            "slot_a {} should be within DA bounds {}",
            slot_a,
            units.len()
        );
        assert_eq!(
            units[slot_a].check, b0 as u32,
            "units[slot_a].check should equal b0"
        );

        // Transition via 'b' (98): slot = b1 + 98 + 1
        let b1 = units[slot_a].base as usize;
        let slot_b = b1 + 98 + 1;
        assert!(
            slot_b < units.len(),
            "slot_b {} should be within DA bounds",
            slot_b
        );
        assert_eq!(
            units[slot_b].check, b1 as u32,
            "units[slot_b].check should equal b1"
        );

        // Terminal: b2 = units[slot_b].base; units[b2].check == b2; units[b2].base == -1000
        let b2 = units[slot_b].base as usize;
        assert!(b2 < units.len(), "b2 {} should be within DA bounds", b2);
        assert_eq!(
            units[b2].check, b2 as u32,
            "terminal check should be self-referential"
        );
        assert_eq!(
            units[b2].base,
            -999 - 1,
            "terminal base should encode value 999"
        );
    }

    /// Test that two keys sharing a prefix both resolve correctly.
    #[test]
    fn test_da_shared_prefix() {
        // "東京" (e69d b1ac ac ac) and "東京都" (+ e9 83 bd)
        let key1 = "東京".as_bytes();
        let key2 = "東京都".as_bytes();

        let mut trie = TrieBuilder::new();
        trie.insert(key1, 7);
        trie.insert(key2, 42);

        let mut da_builder = DaBuilder::new();
        da_builder.build_from_trie(&trie);
        let units = da_builder.finish();

        // Traverse key1 manually and verify terminal
        let b0 = units[0].base as usize;
        let mut b = b0;
        for &byte in key1 {
            let slot = b + byte as usize + 1;
            assert!(
                slot < units.len(),
                "slot out of bounds for key1 byte {byte}"
            );
            assert_eq!(
                units[slot].check, b as u32,
                "check mismatch traversing key1"
            );
            b = units[slot].base as usize;
        }
        // Terminal check
        assert!(b < units.len(), "b out of bounds at terminal of key1");
        assert_eq!(
            units[b].check, b as u32,
            "key1 terminal check must be self-referential"
        );
        assert_eq!(units[b].base, -7 - 1, "key1 value should be 7");

        // Traverse key2 — share prefix with key1, diverge at extra bytes
        let b0 = units[0].base as usize;
        let mut b = b0;
        for &byte in key2 {
            let slot = b + byte as usize + 1;
            assert!(
                slot < units.len(),
                "slot out of bounds for key2 byte {byte}"
            );
            assert_eq!(
                units[slot].check, b as u32,
                "check mismatch traversing key2"
            );
            b = units[slot].base as usize;
        }
        assert!(b < units.len(), "b out of bounds at terminal of key2");
        assert_eq!(
            units[b].check, b as u32,
            "key2 terminal check must be self-referential"
        );
        assert_eq!(units[b].base, -42 - 1, "key2 value should be 42");
    }

    /// Verify write_sysdic header fields beyond magic.
    #[test]
    fn test_write_sysdic_header_fields() {
        let entries = make_entries();
        let dir = std::env::temp_dir();
        let output = dir.join("test_header.dic");
        write_sysdic(&entries, 100, 200, "UTF-8", &output).expect("write should succeed");

        let data = std::fs::read(&output).expect("read output");

        // dict_type = 0 (sys)
        let dict_type = u32::from_le_bytes(data[8..12].try_into().expect("4 bytes"));
        assert_eq!(dict_type, 0);

        // lexicon_size = 4
        let lex_size = u32::from_le_bytes(data[12..16].try_into().expect("4 bytes"));
        assert_eq!(lex_size, 4);

        // left_size = 100
        let left = u32::from_le_bytes(data[16..20].try_into().expect("4 bytes"));
        assert_eq!(left, 100);

        // right_size = 200
        let right = u32::from_le_bytes(data[20..24].try_into().expect("4 bytes"));
        assert_eq!(right, 200);

        // charset = "UTF-8\0..."
        let charset_bytes = &data[40..72];
        let end = charset_bytes.iter().position(|&b| b == 0).unwrap_or(32);
        let charset = std::str::from_utf8(&charset_bytes[..end]).expect("charset utf8");
        assert_eq!(charset, "UTF-8");

        std::fs::remove_file(&output).ok();
    }
}
