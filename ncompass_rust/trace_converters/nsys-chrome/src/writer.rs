//! High-performance streaming JSON writer for Chrome Trace format

use anyhow::{Context, Result};
use gzp::deflate::Gzip;
use gzp::par::compress::{ParCompress, ParCompressBuilder};
use gzp::ZWriter;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};

use crate::models::{ChromeTraceEvent, ChromeTracePhase};

pub const OVERFLOW_NAME_PREFIX: &str = "↳ ";
const OVERFLOW_TID_OFFSET: i64 = 100;

fn compute_overflow_tid(original_tid: &str) -> String {
    if let Ok(tid_num) = original_tid.parse::<i64>() {
        (tid_num + OVERFLOW_TID_OFFSET).to_string()
    } else {
        format!("{}{}", OVERFLOW_NAME_PREFIX, original_tid)
    }
}

struct OverflowState {
    max_end: HashMap<(String, String), f64>,
    thread_names: HashMap<(String, String), String>,
    thread_sort_indices: HashMap<(String, String), i64>,
    overflow_tracks: HashMap<(String, String), (String, String)>,
    moved_events: HashMap<(String, String, OrderedFloat<f64>), String>,
}

impl OverflowState {
    fn new() -> Self {
        Self {
            max_end: HashMap::new(),
            thread_names: HashMap::new(),
            thread_sort_indices: HashMap::new(),
            overflow_tracks: HashMap::new(),
            moved_events: HashMap::new(),
        }
    }

    fn extract_thread_metadata(&mut self, event: &ChromeTraceEvent) {
        if event.ph != ChromeTracePhase::Metadata {
            return;
        }
        let key = (event.pid.clone(), event.tid.clone());
        match event.name.as_str() {
            "thread_name" => {
                if let Some(name) = event.args.get("name").and_then(|v| v.as_str()) {
                    self.thread_names.insert(key, name.to_string());
                }
            }
            "thread_sort_index" => {
                if let Some(idx) = event.args.get("sort_index").and_then(|v| v.as_i64()) {
                    self.thread_sort_indices.insert(key, idx);
                }
            }
            _ => {}
        }
    }

    fn generate_overflow_metadata(&self) -> Vec<ChromeTraceEvent> {
        let mut events = Vec::new();
        for ((pid, overflow_tid), (_, original_tid)) in &self.overflow_tracks {
            let original_key = (pid.clone(), original_tid.clone());

            let overflow_name = if let Some(original_name) = self.thread_names.get(&original_key) {
                format!("{}{}", OVERFLOW_NAME_PREFIX, original_name)
            } else {
                format!("{}{}", OVERFLOW_NAME_PREFIX, original_tid)
            };

            let mut args = HashMap::new();
            args.insert("name".to_string(), serde_json::json!(overflow_name));
            events.push(ChromeTraceEvent::metadata(
                "thread_name".to_string(),
                pid.clone(),
                overflow_tid.clone(),
                args,
            ));

            if let Some(&sort_index) = self.thread_sort_indices.get(&original_key) {
                let mut sort_args = HashMap::new();
                sort_args.insert("sort_index".to_string(), serde_json::json!(sort_index + 1));
                events.push(ChromeTraceEvent::metadata(
                    "thread_sort_index".to_string(),
                    pid.clone(),
                    overflow_tid.clone(),
                    sort_args,
                ));
            }
        }
        events
    }
}

/// Streaming JSON writer for Chrome Trace format
pub struct ChromeTraceWriter;

impl ChromeTraceWriter {
    /// Process a single event for overlap detection and assign to virtual track if needed.
    ///
    /// Perfetto requires strict nesting for events on the same track. Events that partially
    /// overlap (start during previous but end after) get dropped. This function detects
    /// such events and moves them to a virtual overflow track.
    ///
    /// Returns true if the event was moved to an overflow track.
    fn process_event_for_overlap(event: &mut ChromeTraceEvent, state: &mut OverflowState) -> bool {
        if event.ph != ChromeTracePhase::Complete {
            return false;
        }
        let dur = match event.dur {
            Some(d) => d,
            None => return false,
        };

        let ts = event.ts;
        let event_end = ts + dur;
        let original_tid = event.tid.clone();
        let original_key = (event.pid.clone(), original_tid.clone());
        let overflow_tid = compute_overflow_tid(&original_tid);
        let overflow_key = (event.pid.clone(), overflow_tid.clone());

        let orig_max = *state.max_end.get(&original_key).unwrap_or(&f64::NEG_INFINITY);

        if ts >= orig_max || event_end <= orig_max {
            let new_max = orig_max.max(event_end);
            state.max_end.insert(original_key, new_max);
            false
        } else {
            event.tid = overflow_tid.clone();
            let overflow_max = *state.max_end.get(&overflow_key).unwrap_or(&f64::NEG_INFINITY);
            let new_max = overflow_max.max(event_end);
            state.max_end.insert(overflow_key.clone(), new_max);

            state.overflow_tracks.insert(overflow_key, original_key.clone());
            state.moved_events.insert(
                (event.pid.clone(), original_tid, OrderedFloat(ts)),
                event.tid.clone(),
            );
            true
        }
    }

    fn update_flow_event_if_needed(event: &mut ChromeTraceEvent, state: &OverflowState) {
        if event.ph != ChromeTracePhase::FlowStart && event.ph != ChromeTracePhase::FlowFinish {
            return;
        }
        let key = (event.pid.clone(), event.tid.clone(), OrderedFloat(event.ts));
        if let Some(new_tid) = state.moved_events.get(&key) {
            event.tid = new_tid.clone();
        }
    }

    /// Write Chrome Trace events to JSON file
    ///
    /// Automatically handles overlapping events by moving them to virtual overflow
    /// tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    pub fn write(output_path: &str, mut events: Vec<ChromeTraceEvent>) -> Result<()> {
        let file = File::create(output_path)
            .with_context(|| format!("Failed to create output file: {}", output_path))?;
        let mut writer = BufWriter::with_capacity(256 * 1024, file);

        let mut state = OverflowState::new();

        for event in &events {
            state.extract_thread_metadata(event);
        }

        writer.write_all(b"{\"traceEvents\":[\n")?;

        let mut first = true;
        for event in events.iter_mut() {
            Self::process_event_for_overlap(event, &mut state);
            Self::update_flow_event_if_needed(event, &state);

            if !first {
                writer.write_all(b",\n")?;
            } else {
                first = false;
            }
            let json = serde_json::to_vec(&event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;
            writer.write_all(&json)?;
        }

        for metadata_event in state.generate_overflow_metadata() {
            writer.write_all(b",\n")?;
            let json = serde_json::to_vec(&metadata_event)
                .with_context(|| "Failed to serialize overflow metadata event")?;
            writer.write_all(&json)?;
        }

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

        let mut gz_writer: ParCompress<Gzip> = ParCompressBuilder::new().from_writer(file);

        let mut state = OverflowState::new();

        for event in &events {
            state.extract_thread_metadata(event);
        }

        let mut batch_buffer = Vec::with_capacity(300 * 1024);

        batch_buffer.extend_from_slice(b"{\"traceEvents\":[\n");

        let mut first = true;
        for event in events.iter_mut() {
            Self::process_event_for_overlap(event, &mut state);
            Self::update_flow_event_if_needed(event, &state);

            if !first {
                batch_buffer.extend_from_slice(b",\n");
            } else {
                first = false;
            }
            serde_json::to_writer(&mut batch_buffer, &event)
                .with_context(|| format!("Failed to serialize event: {:?}", event))?;

            if batch_buffer.len() >= 256 * 1024 {
                gz_writer.write_all(&batch_buffer)?;
                batch_buffer.clear();
            }
        }

        for metadata_event in state.generate_overflow_metadata() {
            batch_buffer.extend_from_slice(b",\n");
            serde_json::to_writer(&mut batch_buffer, &metadata_event)
                .with_context(|| "Failed to serialize overflow metadata event")?;

            if batch_buffer.len() >= 256 * 1024 {
                gz_writer.write_all(&batch_buffer)?;
                batch_buffer.clear();
            }
        }

        batch_buffer.extend_from_slice(b"\n]}");

        if !batch_buffer.is_empty() {
            gz_writer.write_all(&batch_buffer)?;
        }

        gz_writer
            .finish()
            .with_context(|| "Failed to finish gzip compression")?;

        Ok(())
    }
}
