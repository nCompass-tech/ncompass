//! Event adapter for extracting properties from ChromeTraceEvent

use crate::models::ChromeTraceEvent;

/// Event adapter trait for extracting event properties
pub trait EventAdapter {
    /// Get time range (start, end) from an event in nanoseconds
    fn get_time_range(&self, event: &ChromeTraceEvent) -> Option<(i64, i64)>;

    /// Get correlation ID from an event
    fn get_correlation_id(&self, event: &ChromeTraceEvent) -> Option<i32>;

    /// Get unique event identifier
    fn get_event_id(&self, event: &ChromeTraceEvent) -> EventId;
}

/// Unique identifier for an event (for indexing in overlap maps)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(pub usize);

/// Default event adapter for ChromeTraceEvent from nsys SQLite
pub struct NsysEventAdapter;

impl EventAdapter for NsysEventAdapter {
    fn get_time_range(&self, event: &ChromeTraceEvent) -> Option<(i64, i64)> {
        let start_ns = event
            .args
            .get("start_ns")
            .and_then(|v| v.as_i64())?;
        let end_ns = event
            .args
            .get("end_ns")
            .and_then(|v| v.as_i64())?;
        Some((start_ns, end_ns))
    }

    fn get_correlation_id(&self, event: &ChromeTraceEvent) -> Option<i32> {
        event
            .args
            .get("correlationId")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
    }

    fn get_event_id(&self, event: &ChromeTraceEvent) -> EventId {
        // Use pointer address as unique ID
        EventId(event as *const ChromeTraceEvent as usize)
    }
}

