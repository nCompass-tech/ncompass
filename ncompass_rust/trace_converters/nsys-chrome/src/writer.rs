//! High-performance streaming JSON writer for Chrome Trace format

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::models::ChromeTraceEvent;

/// Streaming JSON writer for Chrome Trace format
pub struct ChromeTraceWriter;

impl ChromeTraceWriter {
    /// Write Chrome Trace events to JSON file
    pub fn write(output_path: &str, events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file); // 256KB buffer

        // Write opening
        writer.write_all(b"{\"traceEvents\":[")?;

        // Write events with commas between them
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                writer.write_all(b",")?;
            }
            let json = serde_json::to_vec(&event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;
            writer.write_all(&json)?;
        }

        // Write closing
        writer.write_all(b"]}")?;
        writer.flush()?;

        Ok(())
    }

    /// Write Chrome Trace events to gzip-compressed JSON file
    pub fn write_gz(output_path: &str, events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        let buf_writer = BufWriter::with_capacity(256 * 1024, file); // 256KB buffer
        let mut gz_writer = GzEncoder::new(buf_writer, Compression::default());

        // Write opening
        gz_writer.write_all(b"{\"traceEvents\":[")?;

        // Write events with commas between them
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                gz_writer.write_all(b",")?;
            }
            let json = serde_json::to_vec(&event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;
            gz_writer.write_all(&json)?;
        }

        // Write closing
        gz_writer.write_all(b"]}")?;
        gz_writer.finish()?;

        Ok(())
    }
}