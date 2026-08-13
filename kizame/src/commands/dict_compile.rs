//! MeCab source → binary dictionary compilation.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! `kizame dict compile` turns a MeCab *source* dictionary — `char.def`,
//! `unk.def`, `matrix.def` and the lexicon `*.csv` files — into the four binary
//! files the mecrab readers map: `char.bin`, `unk.dic`, `matrix.bin` and
//! `sys.dic`.
//!
//! Two properties of the source format drive this module:
//!
//! * **The sources are text, the outputs are not.** Every reader validates an
//!   exact byte layout (`CharDef::parse_bytes` rejects any size other than
//!   `4 + csize*32 + 0xFFFF*4`; `ConnectionMatrix` requires
//!   `4 + lsize*rsize*2`; `SysDic` requires a magic that encodes the file
//!   size). Copying a `.def` file into the output directory therefore produces
//!   a directory that can never be loaded — which is exactly what this command
//!   used to do.
//! * **The sources are not UTF-8.** Stock IPADIC ships in EUC-JP, so every text
//!   file is decoded through `encoding_rs` with the `--charset` encoding before
//!   being parsed. The compiled dictionary always stores UTF-8, so `UTF-8` is
//!   what goes into the `sys.dic` / `unk.dic` charset header.
//!
//! Reference: mecab-0.996 `src/char_property.cpp` (char.def), `src/connector.cpp`
//! (matrix.def), `src/dictionary.cpp` (lexicon / unk.def).

use encoding_rs::Encoding;
use mecrab_builder::{
    CharRange, DicEntry, build_char_bytes, build_matrix_bytes, build_sysdic_bytes,
    build_unkdic_bytes, pack_char_info, set_cost,
};
use std::path::Path;

/// Category names in `mecrab::dict::CharCategory` id order.
///
/// The reader maps a packed `default_type` straight onto this enum
/// (`CharCategory::from(u8)`) and looks unknown-word entries up by these exact
/// names, so a compiled `char.bin` must use these ids — whatever order the
/// source `char.def` happens to declare its categories in.
pub const CANONICAL_CATEGORIES: [&str; 11] = [
    "DEFAULT",
    "SPACE",
    "KANJI",
    "SYMBOL",
    "NUMERIC",
    "ALPHA",
    "HIRAGANA",
    "KATAKANA",
    "KANJINUMERIC",
    "GREEK",
    "CYRILLIC",
];

/// Highest code point in the `char.bin` lookup table (`CharDef::TABLE_SIZE - 1`).
const MAX_CODE_POINT: u32 = 0xFFFE;

/// The `INVOKE GROUP LENGTH` triple of one `char.def` category.
#[derive(Debug, Clone, Copy)]
pub struct CategoryFlags {
    /// Run unknown-word processing even when the lexicon matched.
    pub invoke: bool,
    /// Group a whole run of same-category characters into one candidate.
    pub group: bool,
    /// Emit prefix candidates of 1..=`length` characters (4 bit field).
    pub length: u8,
}

/// A parsed `char.def`.
#[derive(Debug)]
pub struct CharDefSource {
    /// Flags per canonical category id; `None` when the source never declares it.
    pub flags: [Option<CategoryFlags>; 11],
    /// Code-point ranges in declaration order (later entries override earlier ones).
    pub ranges: Vec<CharRange>,
}

/// Counts reported after a successful compile.
pub struct CompileStats {
    /// Lexicon entries written to `sys.dic`.
    pub sys_tokens: usize,
    /// Entries written to `unk.dic`.
    pub unk_tokens: usize,
    /// Connection-matrix dimensions.
    pub matrix_size: (u16, u16),
    /// Byte sizes of the four outputs, in the order sys/matrix/char/unk.
    pub bytes: [usize; 4],
}

/// Resolve a `char.def` category name to its canonical id.
fn canonical_id(name: &str) -> Result<usize, String> {
    CANONICAL_CATEGORIES
        .iter()
        .position(|&c| c == name)
        .ok_or_else(|| {
            format!(
                "unknown character category {name:?}: mecrab's CharCategory only defines {}. \
                 A dictionary using other categories cannot be represented — rename the category \
                 or extend CharCategory first.",
                CANONICAL_CATEGORIES.join(", ")
            )
        })
}

/// Read `path` and decode it with `charset`.
///
/// # Errors
///
/// Returns an error when the file cannot be read, when `charset` is not a label
/// `encoding_rs` knows, or when the bytes are not valid in that encoding.
pub fn decode_file(path: &Path, charset: &str) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    decode_bytes(&bytes, charset, path)
}

/// Decode `bytes` with `charset`, naming `path` in any error.
fn decode_bytes(bytes: &[u8], charset: &str, path: &Path) -> Result<String, String> {
    let encoding = Encoding::for_label(charset.as_bytes())
        .ok_or_else(|| format!("unknown --charset {charset:?} (try utf-8, euc-jp or shift_jis)"))?;

    let (text, _, had_errors) = encoding.decode(bytes);
    if had_errors {
        return Err(format!(
            "{} is not valid {}: it decoded with replacement characters. \
             Stock IPADIC sources are EUC-JP — pass the right --charset.",
            path.display(),
            encoding.name()
        ));
    }
    Ok(text.into_owned())
}

/// Strip a `#` comment and surrounding whitespace from one source line.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(idx) => line[..idx].trim(),
        None => line.trim(),
    }
}

/// Parse one `0xXXXX` / `0xXXXX..0xYYYY` code-point spec.
fn parse_code_point_range(spec: &str, line_no: usize) -> Result<(u32, u32), String> {
    let parse_one = |s: &str| -> Result<u32, String> {
        let hex = s
            .strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))
            .ok_or_else(|| format!("char.def line {line_no}: {s:?} is not a 0x… code point"))?;
        u32::from_str_radix(hex, 16)
            .map_err(|e| format!("char.def line {line_no}: bad code point {s:?}: {e}"))
    };

    match spec.split_once("..") {
        Some((lo, hi)) => {
            let (lo, hi) = (parse_one(lo)?, parse_one(hi)?);
            if lo > hi {
                return Err(format!(
                    "char.def line {line_no}: range {spec:?} runs backwards"
                ));
            }
            Ok((lo, hi))
        }
        None => {
            let cp = parse_one(spec)?;
            Ok((cp, cp))
        }
    }
}

/// Parse a MeCab `char.def`.
///
/// The format is two sections in one file: category definitions
/// (`NAME INVOKE GROUP LENGTH`) and code-point mappings
/// (`0xXXXX[..0xYYYY] CATEGORY [COMPATIBLE_CATEGORY …]`). Lines that start with
/// `0x` are mappings, everything else is a definition; `#` starts a comment.
///
/// The emitted ranges start with a base range that assigns `DEFAULT` to every
/// code point, matching MeCab's rule that an unmapped code point belongs to the
/// mandatory `DEFAULT` category — `mecab-dict-index` writes exactly the same
/// table.
///
/// # Errors
///
/// Returns an error on a malformed line, an unknown category name, or a missing
/// `DEFAULT` category.
pub fn parse_char_def(text: &str) -> Result<CharDefSource, String> {
    let mut flags: [Option<CategoryFlags>; 11] = [None; 11];
    let mut mappings: Vec<(u32, u32, usize, Vec<usize>)> = Vec::new();

    for (idx, raw) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comment(raw);
        if line.is_empty() {
            continue;
        }

        if line.starts_with("0x") || line.starts_with("0X") {
            let mut fields = line.split_whitespace();
            let spec = fields
                .next()
                .ok_or_else(|| format!("char.def line {line_no}: empty mapping"))?;
            let (lo, hi) = parse_code_point_range(spec, line_no)?;

            let names: Vec<&str> = fields.collect();
            let (default_name, compat) = names
                .split_first()
                .ok_or_else(|| format!("char.def line {line_no}: {spec} has no category name"))?;
            let default_id =
                canonical_id(default_name).map_err(|e| format!("char.def line {line_no}: {e}"))?;
            let mut compat_ids = Vec::with_capacity(compat.len());
            for name in compat {
                compat_ids
                    .push(canonical_id(name).map_err(|e| format!("char.def line {line_no}: {e}"))?);
            }
            mappings.push((lo, hi, default_id, compat_ids));
            continue;
        }

        // Category definition: NAME INVOKE GROUP LENGTH
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 4 {
            return Err(format!(
                "char.def line {line_no}: expected 'NAME INVOKE GROUP LENGTH', got {line:?}"
            ));
        }
        let id = canonical_id(fields[0]).map_err(|e| format!("char.def line {line_no}: {e}"))?;
        let num = |s: &str, what: &str| -> Result<u32, String> {
            s.parse::<u32>()
                .map_err(|e| format!("char.def line {line_no}: bad {what} {s:?}: {e}"))
        };
        let invoke = num(fields[1], "INVOKE")? != 0;
        let group = num(fields[2], "GROUP")? != 0;
        let length = num(fields[3], "LENGTH")?;
        if length > 0xF {
            return Err(format!(
                "char.def line {line_no}: LENGTH {length} does not fit the 4 bit field (max 15)"
            ));
        }
        flags[id] = Some(CategoryFlags {
            invoke,
            group,
            length: length as u8,
        });
    }

    let default_flags = flags[0]
        .ok_or_else(|| "char.def does not define the mandatory DEFAULT category".to_string())?;

    // Base range: every code point is DEFAULT until a mapping says otherwise.
    let mut ranges = Vec::with_capacity(mappings.len() + 1);
    ranges.push(CharRange {
        lo: 0,
        hi: MAX_CODE_POINT,
        info: pack_char_info(
            1,
            0,
            default_flags.length,
            default_flags.group,
            default_flags.invoke,
        ),
    });

    for (lo, hi, default_id, compat_ids) in mappings {
        let Some(def) = flags[default_id] else {
            return Err(format!(
                "char.def maps 0x{lo:04X}..0x{hi:04X} to category {} before defining it",
                CANONICAL_CATEGORIES[default_id]
            ));
        };
        let mut type_mask = 1u32 << default_id;
        for id in compat_ids {
            type_mask |= 1u32 << id;
        }
        ranges.push(CharRange {
            lo,
            hi,
            info: pack_char_info(
                type_mask,
                default_id as u8,
                def.length,
                def.group,
                def.invoke,
            ),
        });
    }

    Ok(CharDefSource { flags, ranges })
}

/// Parse one MeCab CSV record into a [`DicEntry`].
///
/// Columns are `surface,left_id,right_id,cost,feature…`; `splitn(5, ',')` keeps
/// every remaining comma inside the feature string, where IPADIC puts the
/// 記号,読点 style commas of its own POS names.
fn parse_csv_record(line: &str, origin: &str, line_no: usize) -> Result<Option<DicEntry>, String> {
    let line = line.trim_end_matches(['\r', '\n']);
    if line.trim().is_empty() {
        return Ok(None);
    }

    let fields: Vec<&str> = line.splitn(5, ',').collect();
    if fields.len() < 5 {
        return Err(format!(
            "{origin} line {line_no}: expected 'surface,left_id,right_id,cost,feature…', got {line:?}"
        ));
    }

    let num = |s: &str, what: &str| -> Result<i64, String> {
        s.trim()
            .parse::<i64>()
            .map_err(|e| format!("{origin} line {line_no}: bad {what} {s:?}: {e}"))
    };

    let left_id = num(fields[1], "left_id")?;
    let right_id = num(fields[2], "right_id")?;
    let wcost = num(fields[3], "cost")?;

    let range = |v: i64, what: &str, lo: i64, hi: i64| -> Result<(), String> {
        if !(lo..=hi).contains(&v) {
            return Err(format!(
                "{origin} line {line_no}: {what} {v} is outside {lo}..={hi}"
            ));
        }
        Ok(())
    };
    range(left_id, "left_id", 0, i64::from(u16::MAX))?;
    range(right_id, "right_id", 0, i64::from(u16::MAX))?;
    range(wcost, "cost", i64::from(i16::MIN), i64::from(i16::MAX))?;

    Ok(Some(DicEntry {
        surface: fields[0].to_string(),
        left_id: left_id as u16,
        right_id: right_id as u16,
        // IPADIC CSVs carry no pos_id column; mecrab only uses it for reporting.
        pos_id: 0,
        wcost: wcost as i16,
        feature: fields[4].to_string(),
    }))
}

/// Parse a lexicon `*.csv` file.
///
/// # Errors
///
/// Returns an error on any malformed record.
pub fn parse_lexicon_csv(text: &str, origin: &str) -> Result<Vec<DicEntry>, String> {
    let mut entries = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        if let Some(entry) = parse_csv_record(line, origin, idx + 1)? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

/// Parse `unk.def`.
///
/// Identical in shape to a lexicon CSV, except that the surface column is a
/// character category name — which must be one mecrab can represent, or the
/// compiled `unk.dic` would hold entries nothing ever looks up.
///
/// # Errors
///
/// Returns an error on a malformed record or an unknown category name.
pub fn parse_unk_def(text: &str) -> Result<Vec<DicEntry>, String> {
    let entries = parse_lexicon_csv(text, "unk.def")?;
    for entry in &entries {
        canonical_id(&entry.surface).map_err(|e| format!("unk.def: {e}"))?;
    }
    if entries.is_empty() {
        return Err("unk.def defines no unknown-word entries".to_string());
    }
    Ok(entries)
}

/// Parse `matrix.def` into `(lsize, rsize, costs)`.
///
/// The first line holds the two dimensions; every later line is
/// `rc_attr lc_attr cost` — the **right**-context id of the preceding word
/// followed by the **left**-context id of the following one, which is the order
/// [`ConnectionMatrix::cost`](mecrab::dict::ConnectionMatrix::cost) takes its
/// arguments in. The flat array therefore uses the reader's own index formula,
/// `costs[rc_attr + lsize * lc_attr]` (MeCab's
/// `matrix_[rcAttr + lsize_ * lcAttr]`), applied through
/// [`mecrab_builder::set_cost`]. Pairs the file omits stay 0.
///
/// The column order was settled empirically: recompiling IPADIC's own
/// `matrix.def` reproduces the `matrix.bin` that `mecab-dict-index` ships
/// byte-for-byte with this formula, and does not with the transposed one.
///
/// # Errors
///
/// Returns an error on a malformed header or record, or an out-of-range id.
pub fn parse_matrix_def(text: &str) -> Result<(u16, u16, Vec<i16>), String> {
    let mut lines = text.lines().enumerate();

    let (header_no, header) = lines
        .by_ref()
        .map(|(i, l)| (i + 1, strip_comment(l)))
        .find(|(_, l)| !l.is_empty())
        .ok_or_else(|| "matrix.def is empty".to_string())?;

    let mut dims = header.split_whitespace();
    let mut dim = |what: &str| -> Result<u16, String> {
        dims.next()
            .ok_or_else(|| format!("matrix.def line {header_no}: missing {what}"))?
            .parse::<u16>()
            .map_err(|e| format!("matrix.def line {header_no}: bad {what}: {e}"))
    };
    let lsize = dim("left_size")?;
    let rsize = dim("right_size")?;

    let cells = lsize as usize * rsize as usize;
    if cells == 0 {
        return Err(format!(
            "matrix.def line {header_no}: dimensions {lsize}×{rsize} describe an empty matrix"
        ));
    }
    let mut costs = vec![0i16; cells];

    for (idx, raw) in lines {
        let line_no = idx + 1;
        let line = strip_comment(raw);
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let mut field = |what: &str| -> Result<i64, String> {
            fields
                .next()
                .ok_or_else(|| format!("matrix.def line {line_no}: missing {what}"))?
                .parse::<i64>()
                .map_err(|e| format!("matrix.def line {line_no}: bad {what}: {e}"))
        };
        let rc_attr = field("right-context id")?;
        let lc_attr = field("left-context id")?;
        let cost = field("cost")?;

        if !(0..i64::from(lsize)).contains(&rc_attr) {
            return Err(format!(
                "matrix.def line {line_no}: right-context id {rc_attr} is outside 0..{lsize}"
            ));
        }
        if !(0..i64::from(rsize)).contains(&lc_attr) {
            return Err(format!(
                "matrix.def line {line_no}: left-context id {lc_attr} is outside 0..{rsize}"
            ));
        }
        if !(i64::from(i16::MIN)..=i64::from(i16::MAX)).contains(&cost) {
            return Err(format!(
                "matrix.def line {line_no}: cost {cost} does not fit an i16"
            ));
        }

        set_cost(
            &mut costs,
            lsize as usize,
            rc_attr as usize,
            lc_attr as usize,
            cost as i16,
        );
    }

    Ok((lsize, rsize, costs))
}

/// Compile a MeCab source dictionary directory into binary form.
///
/// Writes `sys.dic`, `matrix.bin`, `char.bin` and `unk.dic` into `output`
/// (creating it if needed) and copies `dicrc` when present. Nothing is written
/// until every source file has parsed, so a failed compile never leaves a
/// half-built, unloadable dictionary behind.
///
/// # Errors
///
/// Returns an error when a source file is missing, cannot be decoded with
/// `charset`, is malformed, or cannot be written.
pub fn compile_dictionary(
    input: &Path,
    output: &Path,
    charset: &str,
    verbose: bool,
) -> Result<CompileStats, String> {
    // ── Parse every source before writing anything ──────────────────────────
    let char_def = parse_char_def(&decode_file(&input.join("char.def"), charset)?)?;
    let unk_entries = parse_unk_def(&decode_file(&input.join("unk.def"), charset)?)?;
    let (lsize, rsize, costs) =
        parse_matrix_def(&decode_file(&input.join("matrix.def"), charset)?)?;

    // An unk.def entry for a category char.def never declares is unreachable:
    // no code point would ever carry that category id.
    for entry in &unk_entries {
        let id = canonical_id(&entry.surface)?;
        if char_def.flags[id].is_none() {
            return Err(format!(
                "unk.def defines entries for category {} but char.def never declares it",
                entry.surface
            ));
        }
    }

    if verbose {
        println!("  char.def: {} code-point ranges", char_def.ranges.len());
        println!("  unk.def:  {} entries", unk_entries.len());
        println!("  matrix.def: {lsize}×{rsize}");
    }

    let mut csv_files: Vec<std::path::PathBuf> = std::fs::read_dir(input)
        .map_err(|e| format!("cannot read {}: {e}", input.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "csv"))
        .collect();
    csv_files.sort();

    if csv_files.is_empty() {
        return Err(format!(
            "no *.csv lexicon files found in {}",
            input.display()
        ));
    }

    let mut lexicon: Vec<DicEntry> = Vec::new();
    for path in &csv_files {
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let entries = parse_lexicon_csv(&decode_file(path, charset)?, &name)?;
        if verbose {
            println!("  {name}: {} entries", entries.len());
        }
        lexicon.extend(entries);
    }
    if lexicon.is_empty() {
        return Err("the lexicon CSV files contain no entries".to_string());
    }

    // ── Build the four binary images ────────────────────────────────────────
    // Compiled dictionaries always store UTF-8: the sources were decoded above.
    const OUTPUT_CHARSET: &str = "UTF-8";

    let char_bytes = build_char_bytes(&CANONICAL_CATEGORIES, &char_def.ranges)
        .map_err(|e| format!("cannot build char.bin: {e}"))?;
    let matrix_bytes = build_matrix_bytes(lsize, rsize, &costs)
        .map_err(|e| format!("cannot build matrix.bin: {e}"))?;
    let (unk_bytes, unk_stats) = build_unkdic_bytes(
        &unk_entries,
        u32::from(lsize),
        u32::from(rsize),
        OUTPUT_CHARSET,
    )
    .map_err(|e| format!("cannot build unk.dic: {e}"))?;
    let (sys_bytes, sys_stats) = build_sysdic_bytes(
        &lexicon,
        u32::from(lsize),
        u32::from(rsize),
        OUTPUT_CHARSET,
        0,
    )
    .map_err(|e| format!("cannot build sys.dic: {e}"))?;

    // ── Write ───────────────────────────────────────────────────────────────
    std::fs::create_dir_all(output)
        .map_err(|e| format!("cannot create {}: {e}", output.display()))?;

    let write = |name: &str, bytes: &[u8]| -> Result<(), String> {
        let path = output.join(name);
        std::fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
    };
    write("sys.dic", &sys_bytes)?;
    write("matrix.bin", &matrix_bytes)?;
    write("char.bin", &char_bytes)?;
    write("unk.dic", &unk_bytes)?;

    // dicrc is a MeCab runtime config, not a section mecrab reads; copy it so
    // the output directory stays usable with mecab itself.
    let dicrc = input.join("dicrc");
    if dicrc.exists() {
        std::fs::copy(&dicrc, output.join("dicrc"))
            .map_err(|e| format!("cannot copy dicrc: {e}"))?;
    }

    Ok(CompileStats {
        sys_tokens: sys_stats.token_count,
        unk_tokens: unk_stats.token_count,
        matrix_size: (lsize, rsize),
        bytes: [
            sys_bytes.len(),
            matrix_bytes.len(),
            char_bytes.len(),
            unk_bytes.len(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_def_assigns_default_to_unmapped_code_points() {
        let src = parse_char_def("DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n")
            .expect("char.def must parse");
        // Base DEFAULT range plus the one mapping.
        assert_eq!(src.ranges.len(), 2);
        assert_eq!(src.ranges[0].lo, 0);
        assert_eq!(src.ranges[0].hi, MAX_CODE_POINT);
        // DEFAULT: type_mask bit 0, default_type 0, group = 1.
        assert_eq!(src.ranges[0].info, pack_char_info(1, 0, 0, true, false));
        assert_eq!(
            src.ranges[1].info,
            pack_char_info(1 << 1, 1, 0, true, false)
        );
    }

    #[test]
    fn char_def_merges_compatible_categories() {
        let src = parse_char_def(
            "DEFAULT 0 1 0\nKANJI 0 0 2\nKANJINUMERIC 1 1 0\n0x4E00 KANJINUMERIC KANJI\n",
        )
        .expect("char.def must parse");
        let mapping = src.ranges.last().expect("mapping range");
        // type_mask = KANJINUMERIC (bit 8) | KANJI (bit 2); flags from KANJINUMERIC.
        assert_eq!(
            mapping.info,
            pack_char_info((1 << 8) | (1 << 2), 8, 0, true, true)
        );
    }

    #[test]
    fn char_def_rejects_unknown_category() {
        let err = parse_char_def("DEFAULT 0 1 0\nEMOJI 1 1 0\n").expect_err("must reject");
        assert!(
            err.contains("EMOJI"),
            "message must name the category: {err}"
        );
    }

    #[test]
    fn char_def_requires_default() {
        let err = parse_char_def("SPACE 0 1 0\n").expect_err("must reject");
        assert!(
            err.contains("DEFAULT"),
            "message must mention DEFAULT: {err}"
        );
    }

    #[test]
    fn matrix_def_uses_reader_index_formula() {
        let (lsize, rsize, costs) =
            parse_matrix_def("2 3\n0 0 -1\n1 2 7\n").expect("matrix.def must parse");
        assert_eq!((lsize, rsize), (2, 3));
        assert_eq!(costs.len(), 6);
        assert_eq!(costs[0], -1);
        // costs[rc_attr + lsize * lc_attr] = costs[1 + 2*2] = costs[5]
        assert_eq!(costs[5], 7);
        assert_eq!(costs[4], 0, "the transposed cell must stay empty");
    }

    #[test]
    fn matrix_def_rejects_out_of_range_ids() {
        let err = parse_matrix_def("2 2\n5 0 1\n").expect_err("must reject");
        assert!(
            err.contains("right-context id"),
            "message must name the field: {err}"
        );
    }

    #[test]
    fn csv_keeps_commas_inside_the_feature_string() {
        let entries = parse_lexicon_csv(
            "東京,1285,1285,3780,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ\n",
            "test.csv",
        )
        .expect("csv must parse");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].surface, "東京");
        assert_eq!(entries[0].wcost, 3780);
        assert_eq!(
            entries[0].feature,
            "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ"
        );
    }

    #[test]
    fn unk_def_rejects_unknown_category() {
        let err = parse_unk_def("EMOJI,1,1,100,記号,一般,*,*,*,*,*\n").expect_err("must reject");
        assert!(
            err.contains("EMOJI"),
            "message must name the category: {err}"
        );
    }

    #[test]
    fn euc_jp_sources_decode_to_utf8() {
        let (bytes, _, _) = encoding_rs::EUC_JP.encode("東京,1285,1285,3780,名詞,一般");
        let text = decode_bytes(&bytes, "euc-jp", Path::new("test.csv")).expect("must decode");
        assert!(text.starts_with("東京"), "decoded {text:?}");

        // The same bytes read as UTF-8 are mojibake, and must be rejected rather
        // than silently compiled into the dictionary.
        let err = decode_bytes(&bytes, "utf-8", Path::new("test.csv")).expect_err("must reject");
        assert!(err.contains("not valid"), "message: {err}");
    }
}
