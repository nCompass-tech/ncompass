//! Event adapter for extracting properties from ChromeTraceEvent

use log::debug;

use crate::models::{ChromeTraceEvent, ChromeTracePhase};

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
        // Only complete events ("X") have meaningful time ranges for overlap detection
        if event.ph != ChromeTracePhase::Complete {
            debug!(
                "Skipping event '{}': phase {:?} is not Complete",
                event.name, event.ph
            );
            return None;
        }

        let start_ns = match event.args.get("start_ns").and_then(|v| v.as_i64()) {
            Some(v) => v,
            None => {
                debug!(
                    "Skipping event '{}': missing 'start_ns' in args",
                    event.name
                );
                return None;
            }
        };

        let end_ns = match event.args.get("end_ns").and_then(|v| v.as_i64()) {
            Some(v) => v,
            None => {
                debug!(
                    "Skipping event '{}': missing 'end_ns' in args",
                    event.name
                );
                return None;
            }
        };

        Some((start_ns, end_ns))
    }

    fn get_correlation_id(&self, event: &ChromeTraceEvent) -> Option<i32> {
        let corr_id = event
            .args
            .get("correlationId")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32);

        if corr_id.is_none() {
            debug!(
                "Event '{}' has no correlationId in args",
                event.name
            );
        }

        corr_id
    }

    fn get_event_id(&self, event: &ChromeTraceEvent) -> EventId {
        // Use pointer address as unique ID
        EventId(event as *const ChromeTraceEvent as usize)
    }
}

