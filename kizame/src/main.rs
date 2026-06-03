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

mod commands;
#[cfg(feature = "server")]
mod server;
mod tui;

#[cfg(feature = "full")]
use clap::Args;
use clap::{Parser, Subcommand};
#[cfg(feature = "lsp")]
use commands::lsp::LspArgs;
#[cfg(feature = "server")]
use commands::serve::ServeArgs;
use commands::{
    dict::DictCommands, explore::ExploreArgs, parse::ParseArgs, score::ScoreArgs, train::TrainArgs,
    vectors::VectorsCommands,
};
#[cfg(feature = "full")]
use std::path::PathBuf;

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

    /// Start Language Server Protocol (LSP) server
    #[cfg(feature = "lsp")]
    Lsp(LspArgs),

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

    /// Score text: Viterbi cost, segmentation perplexity, entropy
    Score(ScoreArgs),

    /// Train dictionary costs on a MeCab TSV annotated corpus
    Train(TrainArgs),
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

    /// Custom ontology files to import before Wikidata/Wikipedia data.
    ///
    /// Supported formats: `.csv` (surface,uri[,confidence[,type]]),
    /// `.json` (array of OntologyEntry), `.rdf`/`.owl`/`.xml` (RDF/XML).
    /// May be specified multiple times: `--ontology a.csv --ontology b.json`
    #[arg(long = "ontology", value_name = "FILE")]
    ontology_paths: Vec<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Parse(args)) => commands::parse::run_parse(args),
        Some(Commands::Explore(args)) => commands::explore::run_explore(args),
        #[cfg(feature = "server")]
        Some(Commands::Serve(args)) => commands::serve::run_serve(args),
        #[cfg(feature = "lsp")]
        Some(Commands::Lsp(args)) => commands::lsp::run_lsp(args),
        #[cfg(feature = "full")]
        Some(Commands::Build(args)) => run_build(args),
        Some(Commands::Dict { command }) => commands::dict::run_dict(command),
        Some(Commands::Vectors { command }) => commands::vectors::run_vectors(command),
        Some(Commands::Score(args)) => commands::score::run_score(args),
        Some(Commands::Train(args)) => commands::train::run_train(args),
        None => {
            // Default: run parse with top-level args
            commands::parse::run_parse(cli.parse_args)
        }
    }
}

#[cfg(feature = "full")]
fn run_build(args: BuildArgs) -> Result<(), Box<dyn std::error::Error>> {
    use mecrab_builder::BuildConfig;

    eprintln!("KizaMe Builder - Semantic Dictionary Pipeline");
    eprintln!("==============================================");
    eprintln!("Source:         {:?}", args.source);
    eprintln!("Wikidata:       {:?}", args.wikidata);
    eprintln!("Wikipedia:      {:?}", args.wikipedia);
    eprintln!("Ontology files: {}", args.ontology_paths.len());
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
        entity_type_filter: Vec::new(),
        calibration_enabled: true,
        load_existing: None,
        streaming: false,
        dbpedia_path: None,
        online_resolution: false,
        ontology_paths: args.ontology_paths,
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
