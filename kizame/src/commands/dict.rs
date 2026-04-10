//! Dict command - dictionary management subcommands
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use clap::Subcommand;
use std::path::{Path, PathBuf};

#[cfg(feature = "full")]
use mecrab_builder::{DicEntry, write_sysdic};

#[derive(Subcommand)]
pub enum DictCommands {
    /// Initialize dictionary (download and compile IPADIC)
    Init {
        /// Target directory for dictionary
        #[arg(short, long)]
        target: Option<PathBuf>,
    },
    /// Compile CSV dictionary to binary format
    Compile {
        /// Input CSV directory (containing *.csv files)
        #[arg(short = 'i', long)]
        input: PathBuf,

        /// Output directory for compiled dictionary
        #[arg(short = 'o', long)]
        output: PathBuf,

        /// Charset for input files (utf-8, euc-jp, shift_jis)
        #[arg(short = 'c', long, default_value = "utf-8")]
        charset: String,

        /// Verbose output
        #[arg(short = 'v', long)]
        verbose: bool,
    },
    /// Dump dictionary information
    Dump {
        /// Dictionary directory to dump
        #[arg(short = 'd', long)]
        dicdir: PathBuf,

        /// Output vocabulary list (`word_id<TAB>feature`)
        #[arg(long)]
        vocab: bool,

        /// Output vocabulary surface forms (`word_id<TAB>surface`)
        #[arg(long)]
        vocab_surface: bool,
    },
    /// Show dictionary statistics
    Info {
        /// Dictionary directory
        #[arg(short = 'd', long)]
        dicdir: Option<PathBuf>,
    },
}

pub fn run_dict(command: DictCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        DictCommands::Init { target } => run_dict_init(target),
        DictCommands::Compile {
            input,
            output,
            charset,
            verbose,
        } => run_dict_compile(&input, &output, &charset, verbose),
        DictCommands::Dump {
            dicdir,
            vocab,
            vocab_surface,
        } => run_dict_dump(&dicdir, vocab, vocab_surface),
        DictCommands::Info { dicdir } => run_dict_info(dicdir),
    }
}

/// Parse the first line of a text-format `matrix.def` (IPADIC source format) to obtain
/// the `left_size` and `right_size` connection-matrix dimensions.
///
/// The IPADIC source `matrix.def` format is:
/// ```text
/// left_size right_size
/// left_id right_id cost
/// ...
/// ```
///
/// Returns `(left_size, right_size)` as `u32` values suitable for passing to
/// [`mecrab_builder::write_sysdic`].
///
/// # Errors
///
/// Returns an error if the file cannot be read or if the first line is malformed.
fn parse_matrix_def_header(path: &Path) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open matrix.def at {:?}: {}", path, e))?;
    let mut reader = BufReader::new(file);

    let mut first_line = String::new();
    reader
        .read_line(&mut first_line)
        .map_err(|e| format!("Cannot read matrix.def at {:?}: {}", path, e))?;

    let first_line = first_line.trim();
    let mut parts = first_line.split_whitespace();

    let left_str = parts
        .next()
        .ok_or_else(|| format!("matrix.def {:?}: first line is empty", path))?;
    let right_str = parts
        .next()
        .ok_or_else(|| format!("matrix.def {:?}: first line has only one token", path))?;

    let left_size: u32 = left_str.parse().map_err(|e| {
        format!(
            "matrix.def {:?}: cannot parse left_size {:?}: {}",
            path, left_str, e
        )
    })?;
    let right_size: u32 = right_str.parse().map_err(|e| {
        format!(
            "matrix.def {:?}: cannot parse right_size {:?}: {}",
            path, right_str, e
        )
    })?;

    Ok((left_size, right_size))
}

fn run_dict_init(target: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let target = target.unwrap_or_else(|| {
        dirs::data_local_dir()
            .map(|p| p.join("mecrab").join("dic").join("ipadic"))
            .unwrap_or_else(|| PathBuf::from("./ipadic"))
    });

    println!("MeCrab Dictionary Initialization");
    println!("=================================");
    println!();

    // Check if dictionary already exists
    let sys_dic = target.join("sys.dic");
    if sys_dic.exists() {
        println!("Dictionary already exists at: {:?}", target);
        println!();
        println!("To reinstall, remove the directory first:");
        println!("  rm -rf {:?}", target);
        return Ok(());
    }

    // Check standard locations
    let standard_locations = [
        "/var/lib/mecab/dic/ipadic-utf8",
        "/usr/lib/mecab/dic/ipadic-utf8",
        "/usr/local/lib/mecab/dic/ipadic-utf8",
        "/usr/share/mecab/dic/ipadic-utf8",
    ];

    println!("Checking for existing IPADIC installations...");
    for loc in &standard_locations {
        let path = std::path::Path::new(loc);
        if path.join("sys.dic").exists() {
            println!();
            println!("Found IPADIC at: {}", loc);
            println!();
            println!("You can use it directly with:");
            println!("  kizame -d {} parse", loc);
            println!();
            println!("Or create a symlink:");
            println!("  mkdir -p {:?}", target.parent().unwrap_or(&target));
            println!("  ln -s {} {:?}", loc, target);
            return Ok(());
        }
    }

    println!();
    println!("No existing IPADIC found.");
    println!();
    println!("To install IPADIC on your system:");
    println!();
    println!("  # Ubuntu/Debian:");
    println!("  sudo apt install mecab-ipadic-utf8");
    println!();
    println!("  # Fedora/RHEL:");
    println!("  sudo dnf install mecab-ipadic");
    println!();
    println!("  # Arch Linux:");
    println!("  sudo pacman -S mecab-ipadic");
    println!();
    println!("  # macOS (Homebrew):");
    println!("  brew install mecab-ipadic");
    println!();
    println!("After installation, run this command again to verify.");

    Ok(())
}

fn run_dict_compile(
    input: &Path,
    output: &Path,
    charset: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    use std::io::BufReader;
    use std::time::Instant;

    println!("KizaMe Dictionary Compiler");
    println!("==========================");
    println!();
    println!("Input:   {:?}", input);
    println!("Output:  {:?}", output);
    println!("Charset: {}", charset);
    println!();

    let start = Instant::now();

    // Validate input directory
    if !input.is_dir() {
        return Err(format!("Input path is not a directory: {:?}", input).into());
    }

    // Check for required source files
    let required_files = ["char.def", "unk.def", "matrix.def"];
    for file in &required_files {
        let path = input.join(file);
        if !path.exists() {
            return Err(format!("Required file not found: {:?}", path).into());
        }
    }

    // Create output directory
    fs::create_dir_all(output)?;

    // Copy definition files
    if verbose {
        println!("Copying definition files...");
    }
    for file in &["char.def", "unk.def", "matrix.def", "dicrc"] {
        let src = input.join(file);
        let dst = output.join(file);
        if src.exists() {
            fs::copy(&src, &dst)?;
            if verbose {
                println!("  {} -> {:?}", file, dst);
            }
        }
    }

    // Find and process CSV files
    let csv_files: Vec<_> = fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "csv"))
        .collect();

    if csv_files.is_empty() {
        return Err("No CSV files found in input directory".into());
    }

    if verbose {
        println!();
        println!("Found {} CSV files:", csv_files.len());
        for csv in &csv_files {
            println!("  {:?}", csv.file_name().unwrap_or_default());
        }
    }

    // Count entries (pre-scan for progress reporting)
    let mut total_entries = 0usize;
    for csv_path in &csv_files {
        let file = fs::File::open(csv_path)?;
        let reader = BufReader::new(file);
        total_entries += std::io::BufRead::lines(reader).count();
    }

    println!();
    println!("Processing {} entries...", total_entries);

    // Parse left_size / right_size from the text-format matrix.def.
    // The first line of matrix.def (IPADIC source format) is: "left_size right_size"
    let (matrix_left_size, matrix_right_size) = parse_matrix_def_header(&input.join("matrix.def"))?;
    if verbose {
        println!(
            "  matrix.def: left_size={}, right_size={}",
            matrix_left_size, matrix_right_size
        );
    }

    let sys_dic_src = input.join("sys.dic");
    let sys_dic_dst = output.join("sys.dic");

    if sys_dic_src.exists() {
        // Already compiled — just copy
        fs::copy(&sys_dic_src, &sys_dic_dst)?;
        println!("Copied existing sys.dic");
    } else {
        #[cfg(feature = "full")]
        {
            // Real compilation path: parse CSV → DicEntry → write_sysdic
            let mut entries: Vec<DicEntry> = Vec::with_capacity(total_entries);

            for csv_path in &csv_files {
                if verbose {
                    println!(
                        "  Parsing {:?}...",
                        csv_path.file_name().unwrap_or_default()
                    );
                }
                let file = fs::File::open(csv_path)?;
                let reader = BufReader::new(file);

                for (line_num, line_result) in std::io::BufRead::lines(reader).enumerate() {
                    let line = line_result?;
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    // MeCab IPADIC CSV columns:
                    //   0: surface
                    //   1: left_id
                    //   2: right_id
                    //   3: wcost
                    //   4..: feature fields (POS, POS1, POS2, POS3, ctype, cform, base, reading, pron)
                    //
                    // We use splitn(14, ',') so that the last element absorbs any trailing commas
                    // inside feature fields (e.g. compound entries with embedded commas).
                    let fields: Vec<&str> = line.splitn(14, ',').collect();
                    if fields.len() < 4 {
                        if verbose {
                            eprintln!(
                                "  Warning: skipping short line {} in {:?}: {}",
                                line_num + 1,
                                csv_path.file_name().unwrap_or_default(),
                                line
                            );
                        }
                        continue;
                    }

                    let surface = fields[0].to_string();
                    let left_id: u16 = fields[1].trim().parse().unwrap_or(0);
                    let right_id: u16 = fields[2].trim().parse().unwrap_or(0);
                    let wcost: i16 = fields[3].trim().parse().unwrap_or(0);
                    // Remaining fields form the feature string
                    let feature = fields[4..].join(",");

                    entries.push(DicEntry {
                        surface,
                        left_id,
                        right_id,
                        pos_id: 0, // pos_id not present in IPADIC CSV; set to 0
                        wcost,
                        feature,
                    });
                }
            }

            println!("  Loaded {} entries", entries.len());

            // Normalise charset string to upper-case for the header field
            let charset_upper = charset.to_ascii_uppercase();

            let stats = write_sysdic(
                &entries,
                matrix_left_size,
                matrix_right_size,
                &charset_upper,
                &sys_dic_dst,
            )?;

            if verbose {
                println!(
                    "  sys.dic: {} bytes, {} tokens, {} DA units, {} feature bytes",
                    stats.file_size, stats.token_count, stats.da_units, stats.feature_bytes
                );
            }
            println!("Compiled sys.dic ({} entries)", stats.token_count);
        }

        #[cfg(not(feature = "full"))]
        {
            let _ = (sys_dic_dst, matrix_left_size, matrix_right_size);
            println!();
            println!("Note: Real dictionary compilation requires the 'full' feature.");
            println!("Rebuild kizame with:");
            println!("  cargo build -p kizame --features full");
            println!();
            println!("Alternative: Use mecab-dict-index (if installed):");
            println!("  cd {:?}", input);
            println!(
                "  mecab-dict-index -d . -o {:?} -f {} -t utf-8",
                output, charset
            );
            return Err(
                "Dictionary compilation requires --features full or mecab-dict-index".into(),
            );
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("Completed in {:.2?}", elapsed);
    println!();
    println!("Output directory: {:?}", output);
    println!();
    println!("To use this dictionary:");
    println!("  kizame -d {:?} parse", output);

    Ok(())
}

fn run_dict_dump(
    dicdir: &Path,
    vocab: bool,
    vocab_surface: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab::dict::Dictionary;

    let dict = Dictionary::load(dicdir)?;

    // --vocab-surface: output word_id<TAB>surface for every token
    if vocab_surface {
        eprintln!("# Vocabulary surface list");
        eprintln!("# Format: word_id<TAB>surface");
        eprintln!("# Total tokens: {}", dict.sys_dic.token_count());
        eprintln!();

        dict.sys_dic.enumerate_surfaces(|word_id, surface| {
            println!("{}\t{}", word_id, surface);
            true // continue
        });
        return Ok(());
    }

    // If --vocab flag is set, output vocabulary list for Word2Vec training
    if vocab {
        eprintln!("# Vocabulary list for Word2Vec training");
        eprintln!("# Format: word_id<TAB>feature");
        eprintln!("# Total tokens: {}", dict.sys_dic.token_count());
        eprintln!();

        for word_id in 0..dict.sys_dic.token_count() {
            if let Some(token) = dict.sys_dic.token_at(word_id) {
                let feature = dict.sys_dic.get_feature(token);
                println!("{}\t{}", word_id, feature);
            }
        }
        return Ok(());
    }

    // Normal dump mode (human-readable summary)
    println!("Dictionary Dump: {:?}", dicdir);
    println!("================================================================================");
    println!();

    // Basic info
    println!("=== Header Information ===");
    println!("Charset:       {}", dict.charset());
    println!("Lexicon size:  {} entries", dict.size());
    println!();

    // Sample lookups
    println!("=== Sample Entries ===");
    let samples = ["東京", "日本", "私", "食べる", "は", "の", "です"];

    for surface in samples {
        let entries = dict.lookup(surface);
        if !entries.is_empty() {
            println!();
            println!("\"{}\" ({} entries):", surface, entries.len());
            for (i, entry) in entries.iter().take(3).enumerate() {
                println!(
                    "  [{}] cost={:5}, feature={}",
                    i, entry.wcost, entry.feature
                );
            }
            if entries.len() > 3 {
                println!("  ... and {} more", entries.len() - 3);
            }
        }
    }
    println!();

    // Character categories
    println!("=== Character Categories (samples) ===");
    let char_samples = [
        ('あ', "Hiragana"),
        ('ア', "Katakana"),
        ('漢', "Kanji"),
        ('A', "Alpha"),
        ('1', "Numeric"),
        ('　', "Space"),
        ('。', "Symbol"),
    ];

    for (c, expected) in char_samples {
        let info = dict.char_info(c);
        let cat = dict.char_category(c);
        println!(
            "  '{}' ({:10}): category={:?}, invoke={}, group={}, length={}",
            c,
            expected,
            cat,
            info.invoke(),
            info.group(),
            info.length()
        );
    }
    println!();

    // Connection matrix sample
    println!("=== Connection Matrix (sample costs) ===");
    println!("  Format: cost(left_id, right_id)");
    let sample_ids = [0u16, 1, 10, 100, 1000];
    print!("       ");
    for right in &sample_ids {
        print!("{:>7}", right);
    }
    println!();
    for left in &sample_ids {
        print!("  {:>4}:", left);
        for right in &sample_ids {
            let cost = dict.connection_cost(*left, *right);
            print!("{:>7}", cost);
        }
        println!();
    }

    Ok(())
}

fn run_dict_info(dicdir: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let dict = if let Some(path) = dicdir {
        mecrab::dict::Dictionary::load(&path)?
    } else {
        mecrab::dict::Dictionary::default_dictionary()?
    };

    println!("Dictionary Information");
    println!("======================");
    println!("Charset:           {}", dict.charset());
    println!("Lexicon size:      {} entries", dict.size());
    println!("Overlay size:      {} entries", dict.overlay_size());

    Ok(())
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify basic IPADIC CSV line parsing into its constituent fields.
    #[test]
    fn test_parse_ipadic_csv_line() {
        let line = "東京,1285,1285,1438,名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ";
        let fields: Vec<&str> = line.splitn(14, ',').collect();

        assert_eq!(fields[0], "東京", "surface should be 東京");
        assert_eq!(
            fields[1].parse::<u16>().unwrap(),
            1285u16,
            "left_id should be 1285"
        );
        assert_eq!(
            fields[2].parse::<u16>().unwrap(),
            1285u16,
            "right_id should be 1285"
        );
        assert_eq!(
            fields[3].parse::<i16>().unwrap(),
            1438i16,
            "wcost should be 1438"
        );
        assert_eq!(
            &fields[4..].join(","),
            "名詞,固有名詞,地域,一般,*,*,東京,トウキョウ,トウキョウ",
            "feature string should join fields 4.."
        );
    }

    /// Verify that a line with a negative wcost parses correctly.
    #[test]
    fn test_parse_ipadic_csv_line_negative_cost() {
        let line = "は,9,9,-1000,助詞,係助詞,*,*,*,*,は,ハ,ワ";
        let fields: Vec<&str> = line.splitn(14, ',').collect();

        assert_eq!(fields[0], "は");
        assert_eq!(fields[1].trim().parse::<u16>().unwrap_or(0), 9u16);
        assert_eq!(fields[3].trim().parse::<i16>().unwrap_or(0), -1000i16);
        assert_eq!(&fields[4..].join(","), "助詞,係助詞,*,*,*,*,は,ハ,ワ");
    }

    /// Verify that a line with fewer than 4 fields is detected as malformed.
    #[test]
    fn test_parse_ipadic_csv_too_few_fields() {
        let line = "surface,1,2";
        let fields: Vec<&str> = line.splitn(14, ',').collect();
        assert!(fields.len() < 4, "should have fewer than 4 fields");
    }

    /// Verify that `parse_matrix_def_header` correctly parses a well-formed
    /// matrix.def first line.
    #[test]
    fn test_parse_matrix_def_header() {
        use std::io::Write;

        let dir = std::env::temp_dir();
        let path = dir.join("test_matrix_header.def");
        {
            let mut f = std::fs::File::create(&path).expect("create test file");
            writeln!(f, "1316 1316").expect("write");
            writeln!(f, "0 0 0").expect("write data line");
        }

        let (left, right) = parse_matrix_def_header(&path).expect("parse should succeed");
        assert_eq!(left, 1316u32, "left_size should be 1316");
        assert_eq!(right, 1316u32, "right_size should be 1316");

        std::fs::remove_file(&path).ok();
    }

    /// Verify that `parse_matrix_def_header` returns an error for a missing file.
    #[test]
    fn test_parse_matrix_def_header_missing_file() {
        let path = std::path::Path::new("/nonexistent/path/matrix.def");
        let result = parse_matrix_def_header(path);
        assert!(result.is_err(), "should return error for missing file");
    }

    /// Verify that an empty feature section (only 4 fields) yields an empty feature string.
    #[test]
    fn test_parse_ipadic_csv_no_feature_fields() {
        let line = "word,10,20,300";
        let fields: Vec<&str> = line.splitn(14, ',').collect();
        assert_eq!(fields.len(), 4);
        let feature = fields[4..].join(",");
        assert_eq!(feature, "", "empty feature for 4-field line");
    }
}
