//! CLI for nsys to Chrome Trace converter

use clap::Parser;
use nsys_chrome::{convert_file_gz, ConversionOptions};
use std::path::Path;
use std::process::Command;

#[derive(Parser)]
#[command(
    name = "nsys-chrome",
    about = "Convert nsys reports to Chrome Trace format",
    version
)]
struct Args {
    /// Input file path (.nsys-rep or .sqlite)
    #[arg(value_name = "INPUT")]
    input: String,

    /// Output file path (.json or .json.gz)
    #[arg(short = 'o', long = "output", value_name = "OUTPUT")]
    output: String,

    /// Activity types to include
    #[arg(
        short = 't',
        long = "types",
        value_delimiter = ',',
        default_values = &["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"]
    )]
    activity_types: Vec<String>,

    /// NVTX event name prefixes to filter (comma-separated)
    #[arg(long = "nvtx-prefix", value_delimiter = ',')]
    nvtx_prefix: Option<Vec<String>>,

    /// Include metadata events (process/thread names)
    #[arg(long = "metadata", default_value = "true")]
    include_metadata: bool,

    /// Keep intermediate SQLite file (if converting from .nsys-rep)
    #[arg(long = "keep-sqlite")]
    keep_sqlite: bool,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging from RUST_LOG environment variable
    // This is inherited from the parent process when called via subprocess
    env_logger::init();

    let args = Args::parse();

    // Determine if we need to convert .nsys-rep to SQLite first
    let input_path = Path::new(&args.input);
    let sqlite_path: String;
    let temp_sqlite: Option<tempfile::TempPath>;

    if args.input.ends_with(".nsys-rep") {
        // Convert .nsys-rep to SQLite using nsys CLI
        let sqlite_output = if args.keep_sqlite {
            input_path.with_extension("sqlite")
        } else {
            let temp_dir = tempfile::Builder::new()
                .prefix("nsys-chrome-")
                .suffix(".sqlite")
                .tempfile()?;
            temp_dir.path().to_path_buf()
        };

        eprintln!("Converting .nsys-rep to SQLite...");
        let status = Command::new("nsys")
            .args([
                "export",
                "--type",
                "sqlite",
                "--force-overwrite",
                "true",
                "-o",
                sqlite_output.to_str().unwrap(),
                &args.input,
            ])
            .status()?;

        if !status.success() {
            anyhow::bail!("nsys export failed");
        }

        if args.keep_sqlite {
            sqlite_path = sqlite_output.to_str().unwrap().to_string();
            temp_sqlite = None;
        } else {
            // Create temp file to hold the path
            let temp = tempfile::Builder::new()
                .prefix("nsys-chrome-")
                .suffix(".sqlite")
                .tempfile()?;
            sqlite_path = sqlite_output.to_str().unwrap().to_string();
            temp_sqlite = Some(temp.into_temp_path());
        }
    } else {
        sqlite_path = args.input.clone();
        temp_sqlite = None;
    }

    // Build conversion options
    let options = ConversionOptions {
        activity_types: args.activity_types,
        nvtx_event_prefix: args.nvtx_prefix,
        nvtx_color_scheme: Default::default(),
        include_metadata: args.include_metadata,
    };

    // Convert to Chrome Trace
    eprintln!("Converting to Chrome Trace format...");
    convert_file_gz(&sqlite_path, &args.output, Some(options))?;

    // Clean up temp file if needed
    drop(temp_sqlite);

    eprintln!("✓ Conversion complete: {}", args.output);
    Ok(())
}

