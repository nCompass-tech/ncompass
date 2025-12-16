//! High-performance streaming JSON writer for Chrome Trace format

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;
use gzp::deflate::Gzip;
use gzp::par::compress::{ParCompress, ParCompressBuilder};
use gzp::ZWriter;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::models::{ChromeTraceEvent, ChromeTracePhase};

/// Unicode arrow prefix for overflow tracks (U+21B3)
pub const OVERFLOW_PREFIX: &str = "↳ ";

/// Streaming JSON writer for Chrome Trace format
pub struct ChromeTraceWriter;

impl ChromeTraceWriter {
    /// Process a single event for overlap detection and assign to virtual track if needed.
    ///
    /// Perfetto requires strict nesting for events on the same track. Events that partially
    /// overlap (start during previous but end after) get dropped. This function detects
    /// such events and moves them to a virtual overflow track.
    ///
    /// Returns the (potentially modified) event.
    fn process_event_for_overlap(
        event: &mut ChromeTraceEvent,
        max_end: &mut HashMap<(String, String), f64>,
    ) {
        // Only process Complete events (phase X) with duration
        if event.ph != ChromeTracePhase::Complete {
            return;
        }
        let dur = match event.dur {
            Some(d) => d,
            None => return,
        };

        let ts = event.ts;
        let event_end = ts + dur;
        let original_key = (event.pid.clone(), event.tid.clone());
        let overflow_tid = format!("{}{}", OVERFLOW_PREFIX, event.tid);
        let overflow_key = (event.pid.clone(), overflow_tid.clone());

        let orig_max = *max_end.get(&original_key).unwrap_or(&f64::NEG_INFINITY);

        // Check if event fits on original track:
        // - No overlap (starts after previous ends): ts >= orig_max
        // - Fully nested (ends before previous ends): event_end <= orig_max
        if ts >= orig_max || event_end <= orig_max {
            // Keep on original track
            let new_max = orig_max.max(event_end);
            max_end.insert(original_key, new_max);
        } else {
            // Partial overlap - move to overflow track
            event.tid = overflow_tid;
            let overflow_max = *max_end.get(&overflow_key).unwrap_or(&f64::NEG_INFINITY);
            let new_max = overflow_max.max(event_end);
            max_end.insert(overflow_key, new_max);
        }
    }

    /// Write Chrome Trace events to JSON file
    ///
    /// Automatically handles overlapping events by moving them to virtual overflow
    /// tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    pub fn write(output_path: &str, mut events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file); // 256KB buffer

        // Track max end time per (pid, tid) for overlap detection
        let mut max_end: HashMap<(String, String), f64> = HashMap::new();

        // Write opening with newline
        writer.write_all(b"{\"traceEvents\":[\n")?;

        // Write events with commas between them
        // Each event on its own line to avoid Perfetto parser issues with very long lines
        for (i, event) in events.iter_mut().enumerate() {
            // Process event for overlap and potentially assign to overflow track
            Self::process_event_for_overlap(event, &mut max_end);

            if i > 0 {
                writer.write_all(b",\n")?;
            }
            let json = serde_json::to_vec(&event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;
            writer.write_all(&json)?;
        }

        // Write closing with newline
        writer.write_all(b"\n]}")?;
        writer.flush()?;

        Ok(())
    }

    /// Write Chrome Trace events to gzip-compressed JSON file with parallel compression
    ///
    /// Uses pigz-style parallel gzip compression for significantly faster writes
    /// on multi-core systems. Output is standard gzip format.
    ///
    /// Automatically handles overlapping events by moving them to virtual overflow
    /// tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    pub fn write_gz(output_path: &str, mut events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;

        // Create parallel gzip encoder (pigz-style)
        // Uses all available CPU cores by default
        let mut gz_writer: ParCompress<Gzip> = ParCompressBuilder::new()
            .from_writer(file);

        // Track max end time per (pid, tid) for overlap detection
        let mut max_end: HashMap<(String, String), f64> = HashMap::new();

        // Batch buffer to reduce the number of write calls to encoder
        let mut batch_buffer = Vec::with_capacity(300 * 1024); // 256KB batch +
                                                                                 // Overhead

        // Write opening with newline
        batch_buffer.extend_from_slice(b"{\"traceEvents\":[\n");

        // Write events with commas between them, batching to reduce encoder overhead
        // Each event on its own line to avoid Perfetto parser issues with very long lines
        for (i, event) in events.iter_mut().enumerate() {
            // Process event for overlap and potentially assign to overflow track
            Self::process_event_for_overlap(event, &mut max_end);

            if i > 0 {
                batch_buffer.extend_from_slice(b",\n");
            }
            // Writing to Vec is fast (just memory copies)
            serde_json::to_writer(&mut batch_buffer, &event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;

            // Flush batch to encoder when it gets large enough (256KB threshold)
            if batch_buffer.len() >= 256 * 1024 {
                gz_writer.write_all(&batch_buffer)?;
                batch_buffer.clear();
            }
        }

        // Write closing with newline
        batch_buffer.extend_from_slice(b"\n]}");

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
