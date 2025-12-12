//! High-performance converter from nsys SQLite to Chrome Trace format
//!
//! This library provides functionality to convert NVIDIA Nsight Systems (nsys)
//! SQLite exports to Chrome Trace JSON format (Perfetto-compatible).

pub mod converter;
pub mod linker;
pub mod mapping;
pub mod models;
pub mod parsers;
pub mod schema;
pub mod writer;

pub use converter::NsysChromeConverter;
pub use models::{ChromeTraceEvent, ConversionOptions};
pub use writer::ChromeTraceWriter;

/// Convert nsys SQLite file to Chrome Trace JSON
pub fn convert_file(
    sqlite_path: &str,
    output_path: &str,
    options: Option<ConversionOptions>,
) -> anyhow::Result<()> {
    let converter = NsysChromeConverter::new(sqlite_path, options)?;
    let events = converter.convert()?;
    ChromeTraceWriter::write(output_path, events)?;
    Ok(())
}

/// Convert nsys SQLite to gzip-compressed Chrome Trace JSON
pub fn convert_file_gz(
    sqlite_path: &str,
    output_path: &str,
    options: Option<ConversionOptions>,
) -> anyhow::Result<()> {
    let converter = NsysChromeConverter::new(sqlite_path, options)?;
    let events = converter.convert()?;
    ChromeTraceWriter::write_gz(output_path, events)?;
    Ok(())
}

