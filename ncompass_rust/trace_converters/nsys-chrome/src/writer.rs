//! High-performance streaming JSON writer for Chrome Trace format

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use gzp::deflate::Gzip;
use gzp::par::compress::{ParCompress, ParCompressBuilder};
use gzp::ZWriter;
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

    /// Write Chrome Trace events to gzip-compressed JSON file (single-threaded)
    #[allow(dead_code)]
    pub fn write_gz_single_threaded(output_path: &str, events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        let buf_writer = BufWriter::with_capacity(256 * 1024, file); // 256KB buffer
        let mut gz_writer = GzEncoder::new(buf_writer, Compression::default());

        // Batch buffer to reduce the number of write calls to gzip encoder
        // Each write_all to GzEncoder has overhead (compression state, DEFLATE blocks),
        // so we batch serialized events and flush periodically
        let mut batch_buffer = Vec::with_capacity(256 * 1024); // 256KB batch

        // Write opening
        batch_buffer.extend_from_slice(b"{\"traceEvents\":[");

        // Write events with commas between them, batching to reduce gzip write overhead
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                batch_buffer.push(b',');
            }
            // Writing to Vec is fast (just memory copies), unlike writing to GzEncoder
            serde_json::to_writer(&mut batch_buffer, &event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;

            // Flush batch to gzip when it gets large enough (128KB threshold)
            if batch_buffer.len() >= 128 * 1024 {
                gz_writer.write_all(&batch_buffer)?;
                batch_buffer.clear();
            }
        }

        // Write closing
        batch_buffer.extend_from_slice(b"]}");

        // Flush remaining buffer
        if !batch_buffer.is_empty() {
            gz_writer.write_all(&batch_buffer)?;
        }

        gz_writer.finish()?;

        Ok(())
    }

    /// Write Chrome Trace events to gzip-compressed JSON file with parallel compression
    ///
    /// Uses pigz-style parallel gzip compression for significantly faster writes
    /// on multi-core systems. Output is standard gzip format.
    pub fn write_gz(output_path: &str, events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;

        // Create parallel gzip encoder (pigz-style)
        // Uses all available CPU cores by default
        let mut gz_writer: ParCompress<Gzip> = ParCompressBuilder::new()
            .from_writer(file);

        // Batch buffer to reduce the number of write calls to encoder
        let mut batch_buffer = Vec::with_capacity(256 * 1024); // 256KB batch

        // Write opening
        batch_buffer.extend_from_slice(b"{\"traceEvents\":[");

        // Write events with commas between them, batching to reduce encoder overhead
        for (i, event) in events.iter().enumerate() {
            if i > 0 {
                batch_buffer.push(b',');
            }
            // Writing to Vec is fast (just memory copies)
            serde_json::to_writer(&mut batch_buffer, &event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;

            // Flush batch to encoder when it gets large enough (128KB threshold)
            if batch_buffer.len() >= 128 * 1024 {
                gz_writer.write_all(&batch_buffer)?;
                batch_buffer.clear();
            }
        }

        // Write closing
        batch_buffer.extend_from_slice(b"]}");

        // Flush remaining buffer
        if !batch_buffer.is_empty() {
            gz_writer.write_all(&batch_buffer)?;
        }

        gz_writer
            .finish()
            .with_context(|| "Failed to finish gzip compression")?;

        Ok(())
    }
}
