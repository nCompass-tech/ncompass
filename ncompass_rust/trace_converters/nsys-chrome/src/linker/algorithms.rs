//! Core algorithms for linking events via correlation IDs

use std::collections::HashMap;

use crate::linker::adapters::{EventAdapter, EventId};
use crate::models::ChromeTraceEvent;

/// Event for the sweep-line algorithm
#[derive(Debug, Clone)]
struct SweepEvent<'a> {
    timestamp: i64,
    event_type: i32, // 1 for start, -1 for end
    origin: EventOrigin,
    event_ref: &'a ChromeTraceEvent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventOrigin {
    Source,
    Target,
}

/// Find overlapping intervals using sweep-line algorithm
///
/// Generic implementation that works with any event format via adapter.
/// Accepts slices of references to avoid cloning.
pub fn find_overlapping_intervals<'a>(
    source_events: &[&'a ChromeTraceEvent],
    target_events: &[&'a ChromeTraceEvent],
    adapter: &dyn EventAdapter,
) -> HashMap<EventId, Vec<&'a ChromeTraceEvent>> {
    // Build index map for source events
    let source_index_map: HashMap<usize, usize> = source_events
        .iter()
        .enumerate()
        .map(|(i, &e)| ((e as *const ChromeTraceEvent as usize), i))
        .collect();

    // Create mixed list of start/end events
    let mut mixed_events = Vec::with_capacity((source_events.len() + target_events.len()) * 2);

    // Add source events as start/end pairs
    for &source_event in source_events {
        if let Some((start, end)) = adapter.get_time_range(source_event) {
            mixed_events.push(SweepEvent {
                timestamp: start,
                event_type: 1,
                origin: EventOrigin::Source,
                event_ref: source_event,
            });
            mixed_events.push(SweepEvent {
                timestamp: end,
                event_type: -1,
                origin: EventOrigin::Source,
                event_ref: source_event,
            });
        }
    }

    // Add target events as start/end pairs
    for &target_event in target_events {
        if let Some((start, end)) = adapter.get_time_range(target_event) {
            mixed_events.push(SweepEvent {
                timestamp: start,
                event_type: 1,
                origin: EventOrigin::Target,
                event_ref: target_event,
            });
            mixed_events.push(SweepEvent {
                timestamp: end,
                event_type: -1,
                origin: EventOrigin::Target,
                event_ref: target_event,
            });
        }
    }

    // Sort by timestamp, then by event type (start=1 before end=-1), then by origin
    // For origin: Source should come before Target at same timestamp so that
    // source becomes active before target start is processed (matching Python behavior)
    mixed_events.sort_by(|a, b| {
        a.timestamp
            .cmp(&b.timestamp)
            .then_with(|| b.event_type.cmp(&a.event_type)) // Reverse to put starts before ends
            .then_with(|| {
                let a_origin = matches!(a.origin, EventOrigin::Source) as u8;
                let b_origin = matches!(b.origin, EventOrigin::Source) as u8;
                b_origin.cmp(&a_origin) // Reverse: Source (1) comes before Target (0)
            })
    });

    // Track active source intervals
    let mut active_source_intervals: Vec<&ChromeTraceEvent> = Vec::new();
    let mut result_by_index: HashMap<usize, Vec<&ChromeTraceEvent>> = HashMap::default();

    for sweep_event in mixed_events {
        if sweep_event.event_type == 1 {
            // Start event
            if sweep_event.origin == EventOrigin::Source {
                active_source_intervals.push(sweep_event.event_ref);
            } else {
                // Target start - add to all currently active source ranges
                for &source_event in &active_source_intervals {
                    let source_idx = source_index_map[&(source_event as *const ChromeTraceEvent as usize)];
                    result_by_index
                        .entry(source_idx)
                        .or_insert_with(Vec::new)
                        .push(sweep_event.event_ref);
                }
            }
        } else {
            // End event
            if sweep_event.origin == EventOrigin::Source {
                // Remove from active intervals
                if let Some(pos) = active_source_intervals
                    .iter()
                    .position(|&e| std::ptr::eq(e, sweep_event.event_ref))
                {
                    active_source_intervals.remove(pos);
                }
            }
        }
    }

    // Convert to mapping by event identifier
    let mut result = HashMap::default();
    for (idx, target_list) in result_by_index {
        let source_event = source_events[idx];
        let event_id = adapter.get_event_id(source_event);
        result.insert(event_id, target_list);
    }

    result
}

/// Build mapping from correlation ID to list of kernels
/// Accepts a slice of references to avoid cloning.
pub fn build_correlation_map<'a>(
    kernel_events: &[&'a ChromeTraceEvent],
    adapter: &dyn EventAdapter,
) -> HashMap<i32, Vec<&'a ChromeTraceEvent>> {
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::default();

    for &kernel_event in kernel_events {
        if let Some(corr_id) = adapter.get_correlation_id(kernel_event) {
            correlation_map
                .entry(corr_id)
                .or_insert_with(Vec::new)
                .push(kernel_event);
        }
    }

    correlation_map
}

/// Aggregate kernel execution times across multiple kernels
///
/// Finds the minimum start time and maximum end time across all kernels.
pub fn aggregate_kernel_times(
    kernels: &[&ChromeTraceEvent],
    adapter: &dyn EventAdapter,
) -> Option<(i64, i64)> {
    let mut kernel_start_time: Option<i64> = None;
    let mut kernel_end_time: Option<i64> = None;

    for &kernel_event in kernels {
        if let Some((kernel_start, kernel_end)) = adapter.get_time_range(kernel_event) {
            kernel_start_time = Some(
                kernel_start_time
                    .map(|t| t.min(kernel_start))
                    .unwrap_or(kernel_start),
            );
            kernel_end_time = Some(
                kernel_end_time
                    .map(|t| t.max(kernel_end))
                    .unwrap_or(kernel_end),
            );
        }
    }

    match (kernel_start_time, kernel_end_time) {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None,
    }
}

/// Find all kernels associated with an annotation event via overlapping API events
pub fn find_kernels_for_annotation<'a>(
    overlapping_api_events: &[&'a ChromeTraceEvent],
    correlation_map: &HashMap<i32, Vec<&'a ChromeTraceEvent>>,
    adapter: &dyn EventAdapter,
) -> Vec<&'a ChromeTraceEvent> {
    let mut found_kernels = Vec::new();

    for &api_event in overlapping_api_events {
        if let Some(corr_id) = adapter.get_correlation_id(api_event) {
            if let Some(kernels) = correlation_map.get(&corr_id) {
                if !kernels.is_empty() {
                    found_kernels.extend(kernels.iter().copied());
                }
            }
        }
    }

    found_kernels
}

