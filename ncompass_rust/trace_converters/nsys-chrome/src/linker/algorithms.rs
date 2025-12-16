//! Core algorithms for linking events via correlation IDs

use std::cmp::Ordering;
use std::collections::HashMap;

use log::debug;

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

impl<'a> Ord for SweepEvent<'a> {
    /// Sort by timestamp, then starts before ends, then source before target.
    /// Uses lazy evaluation via `then_with` to avoid unnecessary comparisons.
    fn cmp(&self, other: &Self) -> Ordering {
        self.timestamp
            .cmp(&other.timestamp)
            .then_with(|| other.event_type.cmp(&self.event_type)) // Reverse: starts (1) before ends (-1)
            .then_with(|| {
                let self_origin = matches!(self.origin, EventOrigin::Source) as u8;
                let other_origin = matches!(other.origin, EventOrigin::Source) as u8;
                other_origin.cmp(&self_origin) // Reverse: Source (1) comes before Target (0)
            })
    }
}

impl<'a> PartialOrd for SweepEvent<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> PartialEq for SweepEvent<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<'a> Eq for SweepEvent<'a> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventOrigin {
    Source,
    Target,
}

/// Append sweep events (start/end pairs) to destination vector.
///
/// Creates two sweep events per input event: one for start, one for end.
/// Uses mutable reference to avoid extra allocations.
fn append_sweep_events<'a>(
    events: &[&'a ChromeTraceEvent],
    origin: EventOrigin,
    adapter: &dyn EventAdapter,
    dest: &mut Vec<SweepEvent<'a>>,
) {
    for &event in events {
        if let Some((start, end)) = adapter.get_time_range(event) {
            dest.push(SweepEvent {
                timestamp: start,
                event_type: 1,
                origin,
                event_ref: event,
            });
            dest.push(SweepEvent {
                timestamp: end,
                event_type: -1,
                origin,
                event_ref: event,
            });
        }
    }
}

/// Process sorted sweep events using sweep-line algorithm.
///
/// Returns mapping from source index to list of overlapping target events.
fn process_sweep_line<'a>(
    sorted_events: &[SweepEvent<'a>],
    source_index_map: &HashMap<usize, usize>,
) -> HashMap<usize, Vec<&'a ChromeTraceEvent>> {
    let mut active_source_intervals: Vec<&ChromeTraceEvent> = Vec::new();
    let mut result_by_index: HashMap<usize, Vec<&ChromeTraceEvent>> = HashMap::default();

    for sweep_event in sorted_events {
        if sweep_event.event_type == 1 {
            // Start event
            if sweep_event.origin == EventOrigin::Source {
                active_source_intervals.push(sweep_event.event_ref);
            } else {
                // Target start - add to all currently active source ranges
                for &source_event in &active_source_intervals {
                    let source_idx =
                        source_index_map[&(source_event as *const ChromeTraceEvent as usize)];
                    result_by_index
                        .entry(source_idx)
                        .or_default()
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

    result_by_index
}

/// Convert index-based results to EventId-based mapping.
fn convert_to_event_id_map<'a>(
    result_by_index: HashMap<usize, Vec<&'a ChromeTraceEvent>>,
    source_events: &[&'a ChromeTraceEvent],
    adapter: &dyn EventAdapter,
) -> HashMap<EventId, Vec<&'a ChromeTraceEvent>> {
    result_by_index
        .into_iter()
        .map(|(idx, target_list)| (adapter.get_event_id(source_events[idx]), target_list))
        .collect()
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

    // Create sweep events with pre-allocated capacity
    let mut mixed_events = Vec::with_capacity((source_events.len() + target_events.len()) * 2);
    append_sweep_events(source_events, EventOrigin::Source, adapter, &mut mixed_events);
    let source_sweep_count = mixed_events.len();
    append_sweep_events(target_events, EventOrigin::Target, adapter, &mut mixed_events);
    let target_sweep_count = mixed_events.len() - source_sweep_count;

    // Log summary of events processed vs skipped
    let source_skipped = source_events.len() * 2 - source_sweep_count;
    let target_skipped = target_events.len() * 2 - target_sweep_count;
    if source_skipped > 0 || target_skipped > 0 {
        debug!(
            "find_overlapping_intervals: skipped {} source events and {} target events without valid time ranges",
            source_skipped / 2,
            target_skipped / 2
        );
    }

    // Sort using Ord implementation (timestamp -> event_type -> origin)
    mixed_events.sort();

    // Process sweep events and convert to final result
    let result_by_index = process_sweep_line(&mixed_events, &source_index_map);
    let result = convert_to_event_id_map(result_by_index, source_events, adapter);

    debug!(
        "find_overlapping_intervals: found {} source events with overlapping targets",
        result.len()
    );

    result
}

/// Build mapping from correlation ID to list of kernels
/// Accepts a slice of references to avoid cloning.
pub fn build_correlation_map<'a>(
    kernel_events: &[&'a ChromeTraceEvent],
    adapter: &dyn EventAdapter,
) -> HashMap<i32, Vec<&'a ChromeTraceEvent>> {
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::default();
    let mut skipped_count = 0;

    for &kernel_event in kernel_events {
        if let Some(corr_id) = adapter.get_correlation_id(kernel_event) {
            correlation_map
                .entry(corr_id)
                .or_insert_with(Vec::new)
                .push(kernel_event);
        } else {
            skipped_count += 1;
        }
    }

    if skipped_count > 0 {
        debug!(
            "build_correlation_map: skipped {} kernel events without correlationId",
            skipped_count
        );
    }

    debug!(
        "build_correlation_map: built map with {} unique correlation IDs from {} kernels",
        correlation_map.len(),
        kernel_events.len() - skipped_count
    );

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
    let mut api_without_corr_id = 0;
    let mut api_without_kernels = 0;

    for &api_event in overlapping_api_events {
        if let Some(corr_id) = adapter.get_correlation_id(api_event) {
            if let Some(kernels) = correlation_map.get(&corr_id) {
                if !kernels.is_empty() {
                    found_kernels.extend(kernels.iter().copied());
                } else {
                    api_without_kernels += 1;
                }
            } else {
                api_without_kernels += 1;
            }
        } else {
            api_without_corr_id += 1;
        }
    }

    if api_without_corr_id > 0 {
        debug!(
            "find_kernels_for_annotation: {} API events had no correlationId",
            api_without_corr_id
        );
    }
    if api_without_kernels > 0 {
        debug!(
            "find_kernels_for_annotation: {} API events had no associated kernels",
            api_without_kernels
        );
    }

    found_kernels
}

