//! KizaMe (刻め!) - CLI for MeCrab morphological analyzer
//!
//! Copyright 2026 COOLJAPAN OU (Team KitaSan)
//!
//! MeCab → KizaMe (刻め = "Chop up!")
//!
//! ## Subcommands
//!
//! - `kizame parse` - Morphological analysis (default)
//! - `kizame explore` - Interactive lattice debugger TUI
//! - `kizame serve` - HTTP API server (requires --features server)
//! - `kizame build` - Build semantic dictionary from Wikidata (requires --features full)
//! - `kizame dict` - Dictionary management

#[cfg(feature = "server")]
mod server;
mod tui;

use clap::{Args, Parser, Subcommand, ValueEnum};
use mecrab::{MeCrab, OutputFormat};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

/// ANSI color codes for terminal output
mod colors {
    pub const RESET: &str = "\x1b[0m";
    #[allow(dead_code)]
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // POS colors
    pub const NOUN: &str = "\x1b[38;5;39m"; // Blue
    pub const VERB: &str = "\x1b[38;5;208m"; // Orange
    pub const ADJ: &str = "\x1b[38;5;118m"; // Green
    pub const PARTICLE: &str = "\x1b[38;5;243m"; // Gray
    pub const AUX: &str = "\x1b[38;5;141m"; // Purple
    pub const SYMBOL: &str = "\x1b[38;5;245m"; // Light gray
    pub const OTHER: &str = "\x1b[38;5;250m"; // White
}

/// Get color for a POS category
fn pos_color(pos: &str) -> &'static str {
    if pos.starts_with("名詞") {
        colors::NOUN
    } else if pos.starts_with("動詞") {
        colors::VERB
    } else if pos.starts_with("形容詞") || pos.starts_with("形状詞") {
        colors::ADJ
    } else if pos.starts_with("助詞") {
        colors::PARTICLE
    } else if pos.starts_with("助動詞") {
        colors::AUX
    } else if pos.starts_with("記号") || pos.starts_with("補助記号") {
        colors::SYMBOL
    } else {
        colors::OTHER
    }
}

#[derive(Parser)]
#[command(name = "kizame")]
#[command(author = "COOLJAPAN OU (Team KitaSan)")]
#[command(version)]
#[command(about = "KizaMe (刻め!) - MeCrab morphological analyzer CLI")]
#[command(
    long_about = "A high-performance morphological analyzer compatible with MeCab.\n\n\
    MeCab → KizaMe (刻め = \"Carve!\")\n\n\
    Run without subcommand for interactive parsing mode."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    parse_args: ParseArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse text (morphological analysis)
    Parse(ParseArgs),

    /// Interactive lattice debugger TUI ("Matrix" mode)
    Explore(ExploreArgs),

    /// Start HTTP API server
    #[cfg(feature = "server")]
    Serve(ServeArgs),

    /// Build semantic dictionary from Wikidata/Wikipedia dumps
    #[cfg(feature = "full")]
    Build(BuildArgs),

    /// Dictionary management
    Dict {
        #[command(subcommand)]
        command: DictCommands,
    },

    /// Vector embeddings management
    Vectors {
        #[command(subcommand)]
        command: VectorsCommands,
    },
}

#[derive(Args)]
struct ExploreArgs {
    /// Text to analyze and explore
    #[arg(required = true)]
    text: String,

    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    dicdir: Option<PathBuf>,

    /// Path to semantic pool file (semantic.bin)
    #[arg(short = 's', long)]
    semantic_pool: Option<PathBuf>,
}

#[cfg(feature = "server")]
#[derive(Args)]
struct ServeArgs {
    /// Server bind address
    #[arg(short = 'H', long, default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(short = 'p', long, default_value = "3000")]
    port: u16,

    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    dicdir: Option<PathBuf>,
}

#[derive(Subcommand)]
enum DictCommands {
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

        /// Output vocabulary list (`word_id<TAB>surface<TAB>feature`)
        #[arg(long)]
        vocab: bool,
    },
    /// Show dictionary statistics
    Info {
        /// Dictionary directory
        #[arg(short = 'd', long)]
        dicdir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum VectorsCommands {
    /// Train Word2Vec embeddings from corpus
    Train {
        /// Input corpus file (space-separated word_ids per line)
        #[arg(short = 'i', long)]
        input: PathBuf,

        /// Output file (word2vec text format)
        #[arg(short = 'o', long)]
        output: PathBuf,

        /// Embedding vector size
        #[arg(long, default_value = "100")]
        size: usize,

        /// Context window size
        #[arg(long, default_value = "5")]
        window: usize,

        /// Number of negative samples
        #[arg(long, default_value = "5")]
        negative: usize,

        /// Minimum word count
        #[arg(long, default_value = "10")]
        min_count: u64,

        /// Subsampling threshold (1e-4 = 0.0001)
        #[arg(long, default_value = "0.0001")]
        sample: f64,

        /// Initial learning rate
        #[arg(long, default_value = "0.025")]
        alpha: f32,

        /// Minimum learning rate
        #[arg(long, default_value = "0.0001")]
        min_alpha: f32,

        /// Number of training epochs
        #[arg(long, default_value = "3")]
        epochs: usize,

        /// Number of threads
        #[arg(long, default_value = "8")]
        threads: usize,

        /// Output format (text or mcv1)
        #[arg(short = 'f', long, default_value = "text")]
        format: String,

        /// For MCV1 output: maximum word_id in IPADIC
        #[arg(long)]
        max_word_id: Option<u32>,
    },
    /// Convert word2vec/fastText format to MCV1 binary format
    Convert {
        /// Input file (word2vec text format or gensim KeyedVectors)
        #[arg(short = 'i', long)]
        input: PathBuf,

        /// Output file (MCV1 binary format)
        #[arg(short = 'o', long)]
        output: PathBuf,

        /// Input format (word2vec-text, gensim)
        #[arg(short = 'f', long, default_value = "word2vec-text")]
        format: String,

        /// Vocabulary file (`word_id<TAB>feature`) from dict dump --vocab
        #[arg(short = 'v', long)]
        vocab: Option<PathBuf>,
    },
    /// Show vector pool information
    Info {
        /// Vector pool file
        #[arg(short = 'v', long)]
        vector_pool: PathBuf,
    },
}

#[derive(Args, Clone)]
struct ParseArgs {
    /// Path to the dictionary directory
    #[arg(short = 'd', long)]
    dicdir: Option<PathBuf>,

    /// Path to user dictionary
    #[arg(short = 'u', long)]
    userdic: Option<PathBuf>,

    /// Path to semantic pool file (semantic.bin)
    #[arg(short = 's', long)]
    semantic_pool: Option<PathBuf>,

    /// Path to vector pool file (vectors.bin)
    #[arg(short = 'v', long)]
    vector_pool: Option<PathBuf>,

    /// Include semantic URIs in output (requires semantic pool)
    #[arg(long)]
    with_semantic: bool,

    /// Include IPA pronunciation in output
    #[arg(long)]
    with_ipa: bool,

    /// Include word embeddings in output (requires vector pool)
    #[arg(long)]
    with_vector: bool,

    /// Output format
    #[arg(short = 'O', long, value_enum, default_value_t = Format::Default)]
    output_format: Format,

    /// Output wakati (space-separated surface forms only)
    #[arg(short = 'w', long)]
    wakati: bool,

    /// Output wakati with word_id instead of surface (for Word2Vec training)
    #[arg(long)]
    wakati_word_id: bool,

    /// N-best output (number of alternative analyses to show)
    #[arg(short = 'n', long)]
    nbest: Option<usize>,

    /// Enable color output (auto-detected for terminals)
    #[arg(short = 'c', long)]
    color: bool,

    /// Disable color output
    #[arg(long)]
    no_color: bool,

    /// Input file (reads from stdin if not specified)
    #[arg(short = 'i', long)]
    input: Option<PathBuf>,

    /// Output file (writes to stdout if not specified)
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,
}

#[cfg(feature = "full")]
#[derive(Args)]
struct BuildArgs {
    /// Source dictionary CSV (e.g., unidic.csv, ipadic.csv)
    #[arg(short = 's', long)]
    source: PathBuf,

    /// Wikidata JSON dump (latest-all.json.gz)
    #[arg(short = 'w', long)]
    wikidata: Option<PathBuf>,

    /// Wikipedia abstract dump
    #[arg(long)]
    wikipedia: Option<PathBuf>,

    /// Output directory for extended dictionary
    #[arg(short = 'o', long)]
    output: PathBuf,

    /// Maximum semantic candidates per word (default: 5)
    #[arg(long, default_value_t = 5)]
    max_candidates: u8,

    /// Number of parallel workers
    #[arg(short = 'j', long)]
    jobs: Option<usize>,

    /// Verbose output
    #[arg(short = 'v', long)]
    verbose: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Default MeCab format
    Default,
    /// Wakati (space-separated)
    Wakati,
    /// Dump all lattice information
    Dump,
    /// JSON output
    Json,
    /// JSON-LD output with semantic URIs
    Jsonld,
    /// Turtle (TTL) RDF format
    Turtle,
    /// N-Triples RDF format
    Ntriples,
    /// N-Quads RDF format
    Nquads,
}

impl From<Format> for OutputFormat {
    fn from(f: Format) -> Self {
        match f {
            Format::Default => OutputFormat::Default,
            Format::Wakati => OutputFormat::Wakati,
            Format::Dump => OutputFormat::Dump,
            Format::Json => OutputFormat::Json,
            Format::Jsonld => OutputFormat::Jsonld,
            Format::Turtle => OutputFormat::Turtle,
            Format::Ntriples => OutputFormat::Ntriples,
            Format::Nquads => OutputFormat::Nquads,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Parse(args)) => run_parse(args),
        Some(Commands::Explore(args)) => run_explore(args),
        #[cfg(feature = "server")]
        Some(Commands::Serve(args)) => run_serve(args),
        #[cfg(feature = "full")]
        Some(Commands::Build(args)) => run_build(args),
        Some(Commands::Dict { command }) => run_dict(command),
        Some(Commands::Vectors { command }) => run_vectors(command),
        None => {
            // Default: run parse with top-level args
            run_parse(cli.parse_args)
        }
    }
}

fn run_explore(args: ExploreArgs) -> Result<(), Box<dyn std::error::Error>> {
    tui::run_explore(&args.text, args.dicdir, args.semantic_pool)
}

#[cfg(feature = "server")]
fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mecrab = MeCrab::builder().dicdir(args.dicdir).build()?;

    let addr = format!("{}:{}", args.host, args.port)
        .parse()
        .expect("Invalid address");

    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(server::run_server(mecrab, addr))
}

fn run_parse(args: ParseArgs) -> Result<(), Box<dyn std::error::Error>> {
    let format = if args.wakati {
        OutputFormat::Wakati
    } else {
        args.output_format.into()
    };

    let mecrab = MeCrab::builder()
        .dicdir(args.dicdir)
        .userdic(args.userdic)
        .semantic_pool(args.semantic_pool)
        .vector_pool(args.vector_pool)
        .with_semantic(args.with_semantic)
        .with_ipa(args.with_ipa)
        .with_vector(args.with_vector)
        .output_format(format)
        .build()?;

    // Determine input source
    let input: Box<dyn BufRead> = match &args.input {
        Some(path) => {
            let file = std::fs::File::open(path)?;
            Box::new(io::BufReader::new(file))
        }
        None => Box::new(io::stdin().lock()),
    };

    // Determine output destination
    let mut output: Box<dyn Write> = match &args.output {
        Some(path) => {
            let file = std::fs::File::create(path)?;
            Box::new(io::BufWriter::new(file))
        }
        None => Box::new(io::stdout().lock()),
    };

    // Determine if we should use colors
    let use_color = if args.no_color {
        false
    } else if args.color {
        true
    } else {
        // Auto-detect: only if stdout is a terminal, no output file, and not json formats
        args.output.is_none()
            && io::stdout().is_terminal()
            && matches!(format, OutputFormat::Default | OutputFormat::Dump)
    };

    // Setup progress bar for large files
    let show_progress = args.input.is_some() && io::stderr().is_terminal();
    let progress = if show_progress {
        let file_size = args
            .input
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0);

        if file_size > 1024 * 1024 {
            // Only show for files > 1MB
            let pb = indicatif::ProgressBar::new(file_size);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                    .unwrap()
                    .progress_chars("#>-")
            );
            Some(pb)
        } else {
            None
        }
    } else {
        None
    };

    let mut bytes_processed = 0u64;

    for line in input.lines() {
        let line = line?;
        bytes_processed += line.len() as u64 + 1; // +1 for newline

        if let Some(ref pb) = progress {
            pb.set_position(bytes_processed);
        }

        if line.is_empty() {
            continue;
        }

        // Check if N-best mode is requested
        if let Some(n) = args.nbest {
            let results = mecrab.parse_nbest(&line, n)?;
            for (i, (result, cost)) in results.iter().enumerate() {
                if use_color {
                    writeln!(
                        output,
                        "{}# {} (cost={}){}",
                        colors::DIM,
                        i + 1,
                        cost,
                        colors::RESET
                    )?;
                    write_colored_result(&mut output, result)?;
                } else {
                    writeln!(output, "# {} (cost={})", i + 1, cost)?;
                    write!(output, "{result}")?;
                }
                if i < results.len() - 1 {
                    writeln!(output)?;
                }
            }
        } else {
            let result = mecrab.parse(&line)?;

            // Special handling for wakati-word-id mode (for Word2Vec training)
            if args.wakati_word_id {
                let word_ids: Vec<String> = result
                    .morphemes
                    .iter()
                    .map(|m| m.word_id.to_string())
                    .collect();
                writeln!(output, "{}", word_ids.join(" "))?;
            } else if use_color && matches!(format, OutputFormat::Default | OutputFormat::Dump) {
                write_colored_result(&mut output, &result)?;
            } else {
                writeln!(output, "{result}")?;
            }
        }
    }

    if let Some(pb) = progress {
        pb.finish_with_message("done");
    }

    output.flush()?;
    Ok(())
}

/// Write an analysis result with ANSI colors
fn write_colored_result<W: Write>(w: &mut W, result: &mecrab::AnalysisResult) -> io::Result<()> {
    for morpheme in &result.morphemes {
        let features: Vec<&str> = morpheme.feature.split(',').collect();
        let pos = features.first().copied().unwrap_or("*");
        let color = pos_color(pos);

        write!(w, "{}{}{}\t", color, morpheme.surface, colors::RESET)?;
        write!(w, "{}{}{}", colors::DIM, morpheme.feature, colors::RESET)?;
        writeln!(w)?;
    }
    writeln!(w, "{}EOS{}", colors::DIM, colors::RESET)
}

#[cfg(feature = "full")]
fn run_build(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab_builder::BuildConfig;

    eprintln!("KizaMe Builder - Semantic Dictionary Pipeline");
    eprintln!("==============================================");
    eprintln!("Source:         {:?}", args.source);
    eprintln!("Wikidata:       {:?}", args.wikidata);
    eprintln!("Wikipedia:      {:?}", args.wikipedia);
    eprintln!("Output:         {:?}", args.output);
    eprintln!("Max candidates: {}", args.max_candidates);
    eprintln!();

    let config = BuildConfig {
        source_csv: args.source,
        wikidata_path: args.wikidata,
        wikipedia_path: args.wikipedia,
        output_dir: args.output,
        max_candidates: args.max_candidates,
        num_workers: args.jobs.unwrap_or(0),
        verbose: args.verbose,
    };

    let result = mecrab_builder::build_dictionary_sync(config)?;

    eprintln!();
    eprintln!("Build Complete!");
    eprintln!("===============");
    eprintln!("Entries processed:      {}", result.entries_processed);
    eprintln!("Entries with semantics: {}", result.entries_with_semantics);
    eprintln!("Total candidates:       {}", result.total_candidates);
    eprintln!("Output files:");
    for file in &result.output_files {
        eprintln!("  - {:?}", file);
    }

    Ok(())
}

fn run_dict(command: DictCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        DictCommands::Init { target } => run_dict_init(target),
        DictCommands::Compile {
            input,
            output,
            charset,
            verbose,
        } => run_dict_compile(&input, &output, &charset, verbose),
        DictCommands::Dump { dicdir, vocab } => run_dict_dump(&dicdir, vocab),
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
    use std::fs;
    use std::io::{BufReader, BufWriter};
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

    // Count entries
    let mut total_entries = 0usize;
    for csv_path in &csv_files {
        let file = fs::File::open(csv_path)?;
        let reader = BufReader::new(file);
        total_entries += std::io::BufRead::lines(reader).count();
    }

    println!();
    println!("Processing {} entries...", total_entries);

    // For now, we'll use a simplified approach:
    // Copy the existing compiled dictionary if it exists, or instruct user to use mecab-dict-index
    let sys_dic_src = input.join("sys.dic");
    let sys_dic_dst = output.join("sys.dic");

    if sys_dic_src.exists() {
        // Already compiled - just copy
        fs::copy(&sys_dic_src, &sys_dic_dst)?;
        println!("Copied existing sys.dic");
    } else {
        // Need to compile - for now, provide instructions
        println!();
        println!("Dictionary compilation requires building the Double-Array Trie.");
        println!();
        println!("Option 1: Use mecab-dict-index (if available):");
        println!("  cd {:?}", input);
        println!(
            "  mecab-dict-index -d . -o {:?} -f {} -t utf-8 *.csv",
            output, charset
        );
        println!();
        println!("Option 2: Use pre-compiled IPADIC:");
        println!("  kizame dict init");
        println!();

        // Create a minimal sys.dic placeholder
        let placeholder = output.join("sys.dic");
        let mut f = BufWriter::new(fs::File::create(&placeholder)?);
        use std::io::Write;
        // Write a minimal header indicating this needs proper compilation
        writeln!(
            f,
            "# Placeholder - run mecab-dict-index to generate proper sys.dic"
        )?;
        f.flush()?;

        println!("Created placeholder sys.dic - requires full compilation for use.");
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

fn run_dict_dump(dicdir: &Path, vocab: bool) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab::dict::Dictionary;

    let dict = Dictionary::load(dicdir)?;

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

fn run_vectors(command: VectorsCommands) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        VectorsCommands::Train {
            input,
            output,
            size,
            window,
            negative,
            min_count,
            sample,
            alpha,
            min_alpha,
            epochs,
            threads,
            format,
            max_word_id,
        } => run_vectors_train(
            &input,
            &output,
            size,
            window,
            negative,
            min_count,
            sample,
            alpha,
            min_alpha,
            epochs,
            threads,
            &format,
            max_word_id,
        ),
        VectorsCommands::Convert {
            input,
            output,
            format,
            vocab,
        } => run_vectors_convert(&input, &output, &format, vocab.as_deref()),
        VectorsCommands::Info { vector_pool } => run_vectors_info(&vector_pool),
    }
}

fn run_vectors_info(vector_pool: &Path) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab::vectors::VectorStore;
    use std::fs::File;
    use std::sync::Arc;

    println!("Vector Pool Information: {:?}", vector_pool);
    println!("================================================================================");
    println!();

    let file = File::open(vector_pool)?;
    let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
    let store = VectorStore::from_mmap(Arc::new(mmap))?;

    println!("Vocab size:  {}", store.vocab_size());
    println!("Dimensions:  {}", store.dim());
    println!(
        "File size:   {} bytes",
        std::fs::metadata(vector_pool)?.len()
    );

    Ok(())
}

fn run_vectors_convert(
    input: &Path,
    output: &Path,
    format: &str,
    vocab: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};

    match format {
        "word2vec-text" => {
            eprintln!("Converting word2vec text format to MCV1...");
            eprintln!();

            // Step 1: Load vocabulary mapping (surface → word_id)
            let surface_to_word_id: HashMap<String, u32> = if let Some(vocab_path) = vocab {
                eprintln!("Loading vocabulary from {:?}...", vocab_path);
                let file = File::open(vocab_path)?;
                let reader = BufReader::new(file);
                let mut map = HashMap::new();

                for line in reader.lines() {
                    let line = line?;
                    if line.starts_with('#') || line.trim().is_empty() {
                        continue;
                    }

                    let parts: Vec<&str> = line.split('\t').collect();
                    if parts.len() >= 2 {
                        let word_id: u32 = parts[0].parse()?;
                        // Extract surface from feature string (index 6)
                        let features: Vec<&str> = parts[1].split(',').collect();
                        if let Some(surface) = features.get(6) {
                            if *surface != "*" {
                                map.insert(surface.to_string(), word_id);
                            }
                        }
                    }
                }

                eprintln!("  Loaded {} surface forms", map.len());
                map
            } else {
                return Err("vocab file required for word2vec-text format".into());
            };

            // Step 2: Read word2vec text format
            eprintln!("Reading word2vec file {:?}...", input);
            let file = File::open(input)?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            // Read header: vocab_size dim
            let header = lines.next().ok_or("Empty word2vec file")??;
            let header_parts: Vec<&str> = header.split_whitespace().collect();
            if header_parts.len() != 2 {
                return Err("Invalid word2vec header format".into());
            }

            let _w2v_vocab_size: usize = header_parts[0].parse()?;
            let dim: usize = header_parts[1].parse()?;
            eprintln!("  Dimensions: {}", dim);

            // Determine max word_id for output vocab_size
            let max_word_id = surface_to_word_id.values().max().copied().unwrap_or(0);
            let vocab_size = (max_word_id + 1) as usize;
            eprintln!(
                "  Output vocab size: {} (max word_id: {})",
                vocab_size, max_word_id
            );

            // Initialize vectors array (all zeros)
            let mut vectors: Vec<f32> = vec![0.0; vocab_size * dim];

            // Read vectors
            let mut mapped_count = 0;
            let mut unmapped_count = 0;

            for line in lines {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }

                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() != dim + 1 {
                    eprintln!("  Warning: skipping malformed line");
                    continue;
                }

                let surface = parts[0];
                let values: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.parse()).collect();

                match values {
                    Ok(vals) => {
                        if let Some(&word_id) = surface_to_word_id.get(surface) {
                            let idx = word_id as usize * dim;
                            if idx + dim <= vectors.len() {
                                vectors[idx..idx + dim].copy_from_slice(&vals);
                                mapped_count += 1;
                            }
                        } else {
                            unmapped_count += 1;
                        }
                    }
                    Err(e) => {
                        eprintln!("  Warning: failed to parse vector for '{}': {}", surface, e);
                    }
                }
            }

            eprintln!();
            eprintln!("Mapping summary:");
            eprintln!("  Mapped:   {} words", mapped_count);
            eprintln!("  Unmapped: {} words (not in IPADIC)", unmapped_count);
            eprintln!();

            // Step 3: Write MCV1 format
            eprintln!("Writing MCV1 format to {:?}...", output);
            let mut file = File::create(output)?;

            // Write header (32 bytes)
            file.write_all(&0x3143564Du32.to_le_bytes())?; // Magic: MCV1
            file.write_all(&(vocab_size as u32).to_le_bytes())?; // Vocab size
            file.write_all(&(dim as u32).to_le_bytes())?; // Dimension
            file.write_all(&0u32.to_le_bytes())?; // Data type: F32
            file.write_all(&[0u8; 16])?; // Reserved

            // Write vectors
            for value in &vectors {
                file.write_all(&value.to_le_bytes())?;
            }

            let file_size = std::fs::metadata(output)?.len();
            eprintln!(
                "  Wrote {} bytes ({} MB)",
                file_size,
                file_size / 1024 / 1024
            );
            eprintln!();
            eprintln!("✓ Conversion complete!");

            Ok(())
        }
        _ => Err(format!("Unsupported format: {}. Supported: word2vec-text", format).into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_vectors_train(
    input: &Path,
    output: &Path,
    size: usize,
    window: usize,
    negative: usize,
    min_count: u64,
    sample: f64,
    alpha: f32,
    min_alpha: f32,
    epochs: usize,
    threads: usize,
    format: &str,
    max_word_id: Option<u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab_word2vec::Word2VecBuilder;

    eprintln!("Training Word2Vec model...");
    eprintln!("  Input: {:?}", input);
    eprintln!("  Output: {:?}", output);
    eprintln!("  Vector size: {}", size);
    eprintln!("  Window: {}", window);
    eprintln!("  Negative samples: {}", negative);
    eprintln!("  Min count: {}", min_count);
    eprintln!("  Sample: {}", sample);
    eprintln!("  Alpha: {} → {}", alpha, min_alpha);
    eprintln!("  Epochs: {}", epochs);
    eprintln!("  Threads: {}", threads);
    eprintln!();

    // Build model
    let mut model = Word2VecBuilder::new()
        .vector_size(size)
        .window_size(window)
        .negative_samples(negative)
        .min_count(min_count)
        .sample(sample)
        .alpha(alpha)
        .min_alpha(min_alpha)
        .epochs(epochs)
        .threads(threads)
        .build_from_corpus(input)?;

    // Train
    model.train_from_file(input)?;

    // Save
    match format {
        "text" | "word2vec-text" => {
            eprintln!("\nSaving word2vec text format...");
            model.save_text(output)?;
        }
        "mcv1" => {
            eprintln!("\nSaving MCV1 binary format...");
            let max_id = max_word_id.ok_or("--max-word-id required for MCV1 format")?;
            model.save_mcv1(output, max_id)?;
        }
        _ => {
            return Err(format!(
                "Unsupported output format: {}. Supported: text, mcv1",
                format
            )
            .into());
        }
    }

    eprintln!();
    eprintln!("✓ Training complete!");

    Ok(())
}
