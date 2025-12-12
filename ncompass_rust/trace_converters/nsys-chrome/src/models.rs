//! Core data models for Chrome Trace events and conversion options

use serde::Serialize;
use std::collections::HashMap;

/// All valid Chrome Trace event phases
/// Based on Chrome Trace Format spec
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ChromeTracePhase {
    // Duration Events
    #[serde(rename = "B")]
    DurationBegin,
    #[serde(rename = "E")]
    DurationEnd,
    // Complete Events
    #[serde(rename = "X")]
    Complete,
    // Instant Events
    #[serde(rename = "i")]
    Instant,
    // Counter Events
    #[serde(rename = "C")]
    Counter,
    // Async Events
    #[serde(rename = "b")]
    AsyncNestableStart,
    #[serde(rename = "n")]
    AsyncNestableInstant,
    #[serde(rename = "e")]
    AsyncNestableEnd,
    // Flow Events
    #[serde(rename = "s")]
    FlowStart,
    #[serde(rename = "t")]
    FlowStep,
    #[serde(rename = "f")]
    FlowFinish,
    // Sample Events
    #[serde(rename = "P")]
    Sample,
    // Object Events
    #[serde(rename = "N")]
    ObjectCreated,
    #[serde(rename = "O")]
    ObjectSnapshot,
    #[serde(rename = "D")]
    ObjectDestroyed,
    // Metadata Events
    #[serde(rename = "M")]
    Metadata,
    // Memory Dump Events
    #[serde(rename = "V")]
    MemoryDumpGlobal,
    #[serde(rename = "v")]
    MemoryDumpProcess,
    // Mark Events
    #[serde(rename = "R")]
    Mark,
    // Clock Sync Events
    #[serde(rename = "c")]
    ClockSync,
    // Context Events
    #[serde(rename = "(")]
    ContextBegin,
    #[serde(rename = ")")]
    ContextEnd,
}

/// Binding point for flow events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BindingPoint {
    #[serde(rename = "e")]
    Enclosing,
    #[serde(rename = "s")]
    Same,
}

/// Helper type for serializing values that can be string or int
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(untagged)]
pub enum StringOrInt {
    String(String),
    Int(i64),
}

impl From<String> for StringOrInt {
    fn from(s: String) -> Self {
        StringOrInt::String(s)
    }
}

impl From<i64> for StringOrInt {
    fn from(i: i64) -> Self {
        StringOrInt::Int(i)
    }
}

impl From<i32> for StringOrInt {
    fn from(i: i32) -> Self {
        StringOrInt::Int(i as i64)
    }
}

/// Chrome Trace event model with validation
#[derive(Debug, Clone, Serialize)]
pub struct ChromeTraceEvent {
    /// Event name
    pub name: String,
    /// Event phase
    pub ph: ChromeTracePhase,
    /// Timestamp in microseconds
    pub ts: f64,
    /// Process ID (e.g., "Device 0")
    pub pid: String,
    /// Thread ID (e.g., "Stream 1")
    pub tid: String,
    /// Category (e.g., "cuda", "nvtx", "osrt")
    pub cat: String,
    /// Optional metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub args: HashMap<String, serde_json::Value>,
    /// Duration in microseconds (for 'X' events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    /// Color name for visualization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cname: Option<String>,
    /// Flow event ID for linking related events
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<StringOrInt>,
    /// Binding point for flow events: 'e' (enclosing) or 's' (same)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bp: Option<BindingPoint>,
}

impl ChromeTraceEvent {
    /// Create a new Chrome Trace event with required fields
    pub fn new(
        name: String,
        ph: ChromeTracePhase,
        ts: f64,
        pid: String,
        tid: String,
        cat: String,
    ) -> Self {
        Self {
            name,
            ph,
            ts,
            pid,
            tid,
            cat,
            args: HashMap::new(),
            dur: None,
            cname: None,
            id: None,
            bp: None,
        }
    }

    /// Create a complete event (phase 'X') with duration
    pub fn complete(
        name: String,
        ts: f64,
        dur: f64,
        pid: String,
        tid: String,
        cat: String,
    ) -> Self {
        Self {
            name,
            ph: ChromeTracePhase::Complete,
            ts,
            pid,
            tid,
            cat,
            args: HashMap::new(),
            dur: Some(dur),
            cname: None,
            id: None,
            bp: None,
        }
    }

    /// Create a metadata event
    pub fn metadata(name: String, pid: String, tid: String, args: HashMap<String, serde_json::Value>) -> Self {
        Self {
            name,
            ph: ChromeTracePhase::Metadata,
            ts: 0.0,
            pid,
            tid,
            cat: "__metadata".to_string(),
            args,
            dur: None,
            cname: None,
            id: None,
            bp: None,
        }
    }

    /// Create a flow start event
    pub fn flow_start(ts: f64, pid: String, tid: String, id: StringOrInt) -> Self {
        Self {
            name: String::new(),
            ph: ChromeTracePhase::FlowStart,
            ts,
            pid,
            tid,
            cat: "cuda_flow".to_string(),
            args: HashMap::new(),
            dur: None,
            cname: None,
            id: Some(id),
            bp: None,
        }
    }

    /// Create a flow finish event
    pub fn flow_finish(ts: f64, pid: String, tid: String, id: StringOrInt, bp: BindingPoint) -> Self {
        Self {
            name: String::new(),
            ph: ChromeTracePhase::FlowFinish,
            ts,
            pid,
            tid,
            cat: "cuda_flow".to_string(),
            args: HashMap::new(),
            dur: None,
            cname: None,
            id: Some(id),
            bp: Some(bp),
        }
    }

    /// Set event arguments
    pub fn with_args(mut self, args: HashMap<String, serde_json::Value>) -> Self {
        self.args = args;
        self
    }

    /// Add a single argument
    pub fn with_arg<K: Into<String>, V: Into<serde_json::Value>>(mut self, key: K, value: V) -> Self {
        self.args.insert(key.into(), value.into());
        self
    }

    /// Set color name
    pub fn with_color(mut self, cname: String) -> Self {
        self.cname = Some(cname);
        self
    }
}

/// Configuration options for conversion
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Event types to include
    pub activity_types: Vec<String>,
    /// Filter NVTX events by name prefix
    pub nvtx_event_prefix: Option<Vec<String>>,
    /// Color mapping for NVTX events (regex -> color name)
    pub nvtx_color_scheme: HashMap<String, String>,
    /// Include process/thread name metadata events
    pub include_metadata: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            activity_types: vec![
                "kernel".to_string(),
                "nvtx".to_string(),
                "nvtx-kernel".to_string(),
                "cuda-api".to_string(),
                "osrt".to_string(),
                "sched".to_string(),
            ],
            nvtx_event_prefix: None,
            nvtx_color_scheme: HashMap::new(),
            include_metadata: true,
        }
    }
}

/// Utility function to convert nanoseconds to microseconds
#[inline]
pub fn ns_to_us(timestamp_ns: i64) -> f64 {
    timestamp_ns as f64 / 1000.0
}