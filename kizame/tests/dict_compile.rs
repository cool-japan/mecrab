//! End-to-end round trip for `kizame dict compile`.
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! Compiles a small EUC-JP MeCab source dictionary through the real CLI binary,
//! then loads the result with `mecrab::dict::Dictionary::load` and tokenises with
//! it. Both halves matter:
//!
//! * The output directory has to be **loadable** — the compiler used to copy
//!   `char.def` / `matrix.def` / `unk.def` into it as text, where the readers
//!   require exact-size binary images, so every compiled directory failed to
//!   load.
//! * The sources have to be **decoded**, not assumed UTF-8 — stock IPADIC is
//!   EUC-JP, and reading it through `BufRead::lines()` either errored or
//!   produced mojibake features.
//!
//! Requires the `full` feature (the binary writers live in mecrab-builder):
//! `cargo test -p kizame --features full`.
#![cfg(feature = "full")]

use std::path::{Path, PathBuf};
use std::process::Command;

/// A minimal but complete `char.def`, in IPADIC's own shape.
const CHAR_DEF: &str = "\
DEFAULT        0 1 0
SPACE          0 1 0
KANJI          0 0 2
SYMBOL         1 1 0
NUMERIC        1 1 0
ALPHA          1 1 0
HIRAGANA       0 1 2
KATAKANA       1 1 2
KANJINUMERIC   1 1 0
GREEK          1 1 0
CYRILLIC       1 1 0

0x0020 SPACE
0x0021..0x002F SYMBOL
0x0030..0x0039 NUMERIC
0x0041..0x005A ALPHA
0x0061..0x007A ALPHA
0x3000..0x303F SYMBOL
0x3041..0x309F HIRAGANA
0x30A1..0x30FF KATAKANA
0x4E00..0x9FA5 KANJI
0x4E00 KANJINUMERIC KANJI
";

const UNK_DEF: &str = "\
DEFAULT,1,1,10000,記号,一般,*,*,*,*,*
SPACE,1,1,10000,記号,空白,*,*,*,*,*
KANJI,1,1,10000,名詞,一般,*,*,*,*,*
SYMBOL,1,1,10000,記号,一般,*,*,*,*,*
NUMERIC,1,1,10000,名詞,数,*,*,*,*,*
ALPHA,1,1,10000,名詞,一般,*,*,*,*,*
HIRAGANA,1,1,10000,名詞,一般,*,*,*,*,*
KATAKANA,1,1,10000,名詞,一般,*,*,*,*,*
KANJINUMERIC,1,1,10000,名詞,数,*,*,*,*,*
GREEK,1,1,10000,名詞,一般,*,*,*,*,*
CYRILLIC,1,1,10000,名詞,一般,*,*,*,*,*
";

const LEXICON_CSV: &str = "\
東京,1,1,3000,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ
都,1,1,3000,名詞,接尾,地域,*,*,*,都,ト,ト
に,2,2,1000,助詞,格助詞,一般,*,*,*,に,ニ,ニ
住む,3,3,3000,動詞,自立,*,*,五段・マ行,基本形,住む,スム,スム
";

/// 4×4 connection matrix, every transition free.
fn matrix_def() -> String {
    let mut out = String::from("4 4\n");
    for left in 0..4 {
        for right in 0..4 {
            out.push_str(&format!("{left} {right} 0\n"));
        }
    }
    out
}

/// Write `text` to `path` encoded as EUC-JP — the encoding stock IPADIC ships in.
fn write_euc_jp(path: &Path, text: &str) {
    let (bytes, _, had_errors) = encoding_rs::EUC_JP.encode(text);
    assert!(!had_errors, "test fixture must be representable in EUC-JP");
    std::fs::write(path, &bytes).expect("fixture must be writable");
}

/// Create a fresh scratch directory under the system temp dir.
fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("kizame-dict-compile-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    dir
}

/// Lay out an EUC-JP MeCab source dictionary in `dir`.
fn write_sources(dir: &Path) {
    write_euc_jp(&dir.join("char.def"), CHAR_DEF);
    write_euc_jp(&dir.join("unk.def"), UNK_DEF);
    write_euc_jp(&dir.join("matrix.def"), &matrix_def());
    write_euc_jp(&dir.join("lex.csv"), LEXICON_CSV);
}

/// Run `kizame dict compile` and return the finished `Output`.
fn run_compile(input: &Path, output: &Path, charset: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_kizame"))
        .args(["dict", "compile"])
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .args(["--charset", charset])
        .output()
        .expect("kizame binary must run")
}

#[test]
fn compiled_dictionary_loads_and_tokenises() {
    let root = scratch("roundtrip");
    let src = root.join("src");
    let out = root.join("out");
    std::fs::create_dir_all(&src).expect("src dir");
    write_sources(&src);

    let result = run_compile(&src, &out, "euc-jp");
    assert!(
        result.status.success(),
        "compile failed:\n{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );

    // ── The output directory must be binary, and complete ───────────────────
    for name in ["sys.dic", "matrix.bin", "char.bin", "unk.dic"] {
        assert!(out.join(name).exists(), "{name} must be written");
    }
    for name in ["char.def", "unk.def", "matrix.def"] {
        assert!(
            !out.join(name).exists(),
            "{name} is source text and must not be copied into a compiled dictionary"
        );
    }

    // char.bin has one legal size: 4 + categories*32 + 0xFFFF*4.
    let char_len = std::fs::metadata(out.join("char.bin"))
        .expect("char.bin metadata")
        .len();
    assert_eq!(char_len, 4 + 11 * 32 + 0xFFFF * 4, "char.bin size");

    // matrix.bin likewise: 4 + lsize*rsize*2 for the 4×4 matrix above.
    let matrix_len = std::fs::metadata(out.join("matrix.bin"))
        .expect("matrix.bin metadata")
        .len();
    assert_eq!(matrix_len, 4 + 4 * 4 * 2, "matrix.bin size");

    // ── It has to load, and analyse ─────────────────────────────────────────
    let dict = mecrab::dict::Dictionary::load(&out).expect("compiled dictionary must load");
    let mecrab = mecrab::MeCrab::from_dictionary(dict);

    let result = mecrab.parse("東京都に住む").expect("parse must succeed");
    let surfaces: Vec<&str> = result
        .morphemes
        .iter()
        .map(|m| m.surface.as_str())
        .collect();
    assert_eq!(surfaces, vec!["東京", "都", "に", "住む"]);

    // The EUC-JP feature strings must have survived as UTF-8, not mojibake.
    assert_eq!(
        result.morphemes[0].feature,
        "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ"
    );
    assert_eq!(
        result.morphemes[3].feature,
        "動詞,自立,*,*,五段・マ行,基本形,住む,スム,スム"
    );

    // Unknown-word handling has to work too — that needs char.bin and unk.dic
    // to have compiled into a matching pair.
    let unknown = mecrab.parse("アイウエオ").expect("parse must succeed");
    assert_eq!(
        unknown.morphemes.len(),
        1,
        "katakana run groups into one node"
    );
    assert!(
        unknown.morphemes[0].feature.starts_with("名詞,一般"),
        "unknown katakana must use the KATAKANA unk.dic entry, got {:?}",
        unknown.morphemes[0].feature
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn wrong_charset_fails_loudly() {
    let root = scratch("charset");
    let src = root.join("src");
    let out = root.join("out");
    std::fs::create_dir_all(&src).expect("src dir");
    write_sources(&src);

    // The sources are EUC-JP; claiming UTF-8 must fail rather than silently
    // compiling mojibake surfaces and features into the dictionary.
    let result = run_compile(&src, &out, "utf-8");
    assert!(
        !result.status.success(),
        "compiling EUC-JP sources as UTF-8 must fail"
    );
    let message = format!(
        "{}{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(
        message.contains("not valid"),
        "error must explain the decoding failure, got: {message}"
    );
    assert!(
        !out.join("sys.dic").exists(),
        "a failed compile must not leave a partial dictionary behind"
    );

    let _ = std::fs::remove_dir_all(&root);
}
