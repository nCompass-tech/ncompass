//! Main converter class for nsys SQLite to Chrome Trace conversion

use anyhow::{Context, Result};
use rusqlite::Connection;
use serde_json::json;
use std::collections::{HashMap, HashSet};

use crate::linker::link_nvtx_to_kernels;
use crate::mapping::{extract_device_mapping, extract_thread_names, get_all_devices};
use crate::models::{ChromeTraceEvent, ConversionOptions};
use crate::parsers::{
    CUPTIKernelParser, CUPTIRuntimeParser, EventParser, NVTXParser, OSRTParser, ParseContext,
    SchedParser,
};
use crate::schema::detect_event_types;

/// Filter out NVTX events that have been mapped to kernels, keeping only unmapped ones.
/// Consumes the input nvtx_events vector and returns only the unmapped events.
fn filter_unmapped_nvtx_events(
    nvtx_events: Vec<ChromeTraceEvent>,
    mapped_nvtx_identifiers: &HashSet<(i32, i32, i64, String)>,
) -> Vec<ChromeTraceEvent> {
    if mapped_nvtx_identifiers.is_empty() {
        return nvtx_events;
    }

    nvtx_events
        .into_iter()
        .filter(|event| {
            let device_id = event.args.get("deviceId").and_then(|v| v.as_i64());
            let tid = event.args.get("raw_tid").and_then(|v| v.as_i64());
            let start_ns = event.args.get("start_ns").and_then(|v| v.as_i64());

            if let (Some(device_id), Some(tid), Some(start_ns)) = (device_id, tid, start_ns) {
                let event_identifier =
                    (device_id as i32, tid as i32, start_ns, event.name.clone());
                !mapped_nvtx_identifiers.contains(&event_identifier)
            } else {
                true
            }
        })
        .collect()
}

/// Process NVTX-kernel linking if all required events are available.
/// Returns (events_to_add, remaining_nvtx_events).
fn process_nvtx_kernel_linking(
    kernel_events: &[ChromeTraceEvent],
    cuda_api_events: &[ChromeTraceEvent],
    nvtx_events: Vec<ChromeTraceEvent>,
    options: &ConversionOptions,
) -> (Vec<ChromeTraceEvent>, Vec<ChromeTraceEvent>) {
    if kernel_events.is_empty() || cuda_api_events.is_empty() || nvtx_events.is_empty() {
        eprintln!(
            "Warning: nvtx-kernel requested but requires kernel, cuda-api, and nvtx events. Skipping."
        );
        return (Vec::new(), nvtx_events);
    }

    let (nvtx_kernel_events, mapped_nvtx_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, cuda_api_events, kernel_events, options);

    let mut events_to_add = Vec::with_capacity(nvtx_kernel_events.len() + flow_events.len());
    events_to_add.extend(nvtx_kernel_events);
    events_to_add.extend(flow_events);

    // Filter out mapped NVTX events, keep unmapped ones
    let remaining_nvtx = filter_unmapped_nvtx_events(nvtx_events, &mapped_nvtx_identifiers);

    (events_to_add, remaining_nvtx)
}

/// Main converter class for nsys SQLite to Chrome Trace conversion
pub struct NsysChromeConverter {
    conn: Connection,
    options: ConversionOptions,
}

impl NsysChromeConverter {
    /// Create a new converter
    pub fn new(sqlite_path: &str, options: Option<ConversionOptions>) -> Result<Self> {
        let conn = Connection::open(sqlite_path)
            .with_context(|| format!("Failed to open SQLite database: {}", sqlite_path))?;

        let options = options.unwrap_or_default();

        Ok(Self { conn, options })
    }

    /// Load StringIds table into HashMap
    fn load_strings(&self) -> Result<HashMap<i32, String>> {
        let mut strings = HashMap::default();

        // Check if StringIds table exists
        let table_exists: bool = self
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds'")?
            .exists([])?;

        if !table_exists {
            return Ok(strings);
        }

        let mut stmt = self.conn.prepare("SELECT id, value FROM StringIds")?;
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let id: i32 = row.get(0)?;
            let value: String = row.get(1)?;
            strings.insert(id, value);
        }

        Ok(strings)
    }

    /// Detect available event types based on tables
    fn detect_event_types(&self) -> Result<HashSet<String>> {
        detect_event_types(&self.conn)
    }

    /// Parse all events based on options and available tables
    fn parse_all_events(
        &self,
        strings: &HashMap<i32, String>,
        device_map: &HashMap<i32, i32>,
        thread_names: &HashMap<i32, String>,
    ) -> Result<Vec<ChromeTraceEvent>> {
        let mut events = Vec::new();
        let available_activities = self.detect_event_types()?;

        // Filter requested activities by what's actually available
        let requested_activities: HashSet<String> =
            self.options.activity_types.iter().cloned().collect();
        let activities_to_parse: HashSet<String> = requested_activities
            .intersection(&available_activities)
            .cloned()
            .collect();

        // Create parse context
        let context = ParseContext::new(&self.conn, strings, &self.options, device_map, thread_names);

        // Track parsed events for nvtx-kernel linking
        let mut kernel_events = Vec::new();
        let mut cuda_api_events = Vec::new();
        let mut nvtx_events = Vec::new();

        // Parse kernel events
        if activities_to_parse.contains("kernel") {
            let parser = CUPTIKernelParser;
            kernel_events = parser.safe_parse(&context)?;
        }

        // Parse CUDA API events
        if activities_to_parse.contains("cuda-api") {
            let parser = CUPTIRuntimeParser;
            cuda_api_events = parser.safe_parse(&context)?;
        }

        // Parse NVTX events
        if activities_to_parse.contains("nvtx") {
            let parser = NVTXParser;
            nvtx_events = parser.safe_parse(&context)?;
        }

        // Parse nvtx-kernel events (requires linking) - uses references, no cloning
        if activities_to_parse.contains("nvtx-kernel") {
            let (nvtx_kernel_events, remaining_nvtx) = process_nvtx_kernel_linking(
                &kernel_events,
                &cuda_api_events,
                nvtx_events,
                &self.options,
            );
            events.extend(nvtx_kernel_events);
            nvtx_events = remaining_nvtx;
        }

        // Add kernel events (move, not clone)
        events.extend(kernel_events);

        // Add CUDA API events (move, not clone)
        events.extend(cuda_api_events);

        // Add any remaining NVTX events (move, not clone)
        events.extend(nvtx_events);

        // Parse OS runtime events
        if activities_to_parse.contains("osrt") {
            let parser = OSRTParser;
            events.extend(parser.safe_parse(&context)?);
        }

        // Parse scheduling events
        if activities_to_parse.contains("sched") {
            let parser = SchedParser;
            events.extend(parser.safe_parse(&context)?);
        }

        Ok(events)
    }

    /// Add metadata events for process and thread names
    fn add_metadata_events(&self, thread_names: &HashMap<i32, String>) -> Result<Vec<ChromeTraceEvent>> {
        if !self.options.include_metadata {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();

        // Add process name events
        let devices = get_all_devices(&self.conn)?;
        for device_id in &devices {
            let mut args = HashMap::default();
            args.insert("name".to_string(), json!(format!("Device {}", device_id)));

            let event = ChromeTraceEvent::metadata(
                "process_name".to_string(),
                format!("Device {}", device_id),
                String::new(),
                args,
            );
            events.push(event);
        }

        // Add thread name events
        for (&tid, name) in thread_names {
            for device_id in &devices {
                let mut args = HashMap::default();
                args.insert("name".to_string(), json!(name));

                let event = ChromeTraceEvent::metadata(
                    "thread_name".to_string(),
                    format!("Device {}", device_id),
                    format!("Thread {}", tid),
                    args,
                );
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Sort events by timestamp, then pid, then tid
    fn sort_events(mut events: Vec<ChromeTraceEvent>) -> Vec<ChromeTraceEvent> {
        events.sort_by(|a, b| {
            a.ts
                .partial_cmp(&b.ts)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.pid.cmp(&b.pid))
                .then_with(|| a.tid.cmp(&b.tid))
        });
        events
    }

    /// Perform the conversion
    pub fn convert(self) -> Result<Vec<ChromeTraceEvent>> {
        // Load required data
        
        let strings = self.load_strings()?;
        let device_map = extract_device_mapping(&self.conn)?;
        let thread_names = extract_thread_names(&self.conn)?;

        // Parse all events
        let mut events = self.parse_all_events(&strings, &device_map, &thread_names)?;

        // Add metadata events
        if self.options.include_metadata {
            events.extend(self.add_metadata_events(&thread_names)?);
        }

        // Sort events
        events = Self::sort_events(events);

        Ok(events)
    }
}

