//! Dict command - dictionary management subcommands
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)

use clap::Subcommand;
use std::path::{Path, PathBuf};

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

        /// Charset of the source files (euc-jp, utf-8, shift_jis).
        ///
        /// Stock IPADIC sources are EUC-JP, which is also `mecab-dict-index`'s
        /// default. The compiled dictionary always stores UTF-8.
        #[arg(short = 'c', long, default_value = "euc-jp")]
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
    for file in &["char.def", "unk.def", "matrix.def"] {
        let path = input.join(file);
        if !path.exists() {
            return Err(format!("Required file not found: {:?}", path).into());
        }
    }

    compile_sources(input, output, charset, verbose)?;

    println!();
    println!("Completed in {:.2?}", start.elapsed());
    println!();
    println!("Output directory: {:?}", output);
    println!();
    println!("To use this dictionary:");
    println!("  kizame -d {:?} parse", output);

    Ok(())
}

/// Compile the MeCab source files in `input` into binary form in `output`.
#[cfg(feature = "full")]
fn compile_sources(
    input: &Path,
    output: &Path,
    charset: &str,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stats = super::dict_compile::compile_dictionary(input, output, charset, verbose)?;

    println!();
    println!("Compiled:");
    println!(
        "  sys.dic:    {:>12} bytes  ({} tokens)",
        stats.bytes[0], stats.sys_tokens
    );
    println!(
        "  matrix.bin: {:>12} bytes  ({}x{})",
        stats.bytes[1], stats.matrix_size.0, stats.matrix_size.1
    );
    println!("  char.bin:   {:>12} bytes", stats.bytes[2]);
    println!(
        "  unk.dic:    {:>12} bytes  ({} tokens)",
        stats.bytes[3], stats.unk_tokens
    );

    Ok(())
}

/// Without the `full` feature the binary writers are not linked in.
///
/// This fails before writing anything: a partially populated output directory —
/// the `.def` source files copied as-is, as earlier revisions did — is an
/// unloadable dictionary, not a head start.
#[cfg(not(feature = "full"))]
fn compile_sources(
    input: &Path,
    output: &Path,
    charset: &str,
    _verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Dictionary compilation requires the 'full' feature.");
    println!("Rebuild kizame with:");
    println!("  cargo build -p kizame --features full");
    println!();
    println!("Alternative: use mecab-dict-index (if installed):");
    println!("  cd {:?}", input);
    println!(
        "  mecab-dict-index -d . -o {:?} -f {} -t utf-8",
        output, charset
    );
    Err("Dictionary compilation requires --features full or mecab-dict-index".into())
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
