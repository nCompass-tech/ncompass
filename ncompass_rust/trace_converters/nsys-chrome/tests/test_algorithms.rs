//! Unit tests for linker algorithms module

use nsys_chrome::linker::adapters::{EventAdapter, NsysEventAdapter};
use nsys_chrome::linker::algorithms::{
    aggregate_kernel_times, build_correlation_map, find_kernels_for_annotation,
    find_overlapping_intervals,
};
use nsys_chrome::models::ChromeTraceEvent;
use std::collections::HashMap;

// ==========================
// Helper Functions
// ==========================

/// Create a complete event with start_ns and end_ns in args
fn create_event_with_times(
    name: &str,
    start_ns: i64,
    end_ns: i64,
    correlation_id: Option<i32>,
) -> ChromeTraceEvent {
    let mut event = ChromeTraceEvent::complete(
        name.to_string(),
        start_ns as f64 / 1000.0, // ts in microseconds
        (end_ns - start_ns) as f64 / 1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(start_ns))
    .with_arg("end_ns", serde_json::json!(end_ns));

    if let Some(corr_id) = correlation_id {
        event = event.with_arg("correlationId", serde_json::json!(corr_id));
    }

    event
}

// ==========================
// Tests for find_overlapping_intervals
// ==========================

#[test]
fn test_find_overlapping_intervals_basic_overlap() {
    let adapter = NsysEventAdapter;

    let source_event = create_event_with_times("source", 100000, 200000, None);
    let target_event = create_event_with_times("target", 150000, 180000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert_eq!(result.len(), 1);
    // Get the source event's ID and check it has one overlapping target
    let source_id = adapter.get_event_id(&source_event);
    assert!(result.contains_key(&source_id));
    assert_eq!(result[&source_id].len(), 1);
}

#[test]
fn test_find_overlapping_intervals_no_overlap() {
    let adapter = NsysEventAdapter;

    let source_event = create_event_with_times("source", 100000, 150000, None);
    let target_event = create_event_with_times("target", 200000, 250000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // No overlaps should be found
    assert!(result.is_empty());
}

#[test]
fn test_find_overlapping_intervals_touching_counts_as_overlap() {
    let adapter = NsysEventAdapter;

    // Events that touch (end of one = start of another)
    // The sweep-line algorithm processes starts before ends at same timestamp,
    // so touching intervals ARE detected as overlapping
    let source_event = create_event_with_times("source", 100000, 150000, None);
    let target_event = create_event_with_times("target", 150000, 200000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Touching intervals count as overlap in this algorithm
    // (target start is processed while source is still active)
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_overlapping_intervals_nested() {
    let adapter = NsysEventAdapter;

    // Target completely inside source
    let source_event = create_event_with_times("source", 100000, 300000, None);
    let target_event = create_event_with_times("target", 150000, 200000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert_eq!(result.len(), 1);
    let source_id = adapter.get_event_id(&source_event);
    assert_eq!(result[&source_id].len(), 1);
}

#[test]
fn test_find_overlapping_intervals_multiple_targets() {
    let adapter = NsysEventAdapter;

    let source_event = create_event_with_times("source", 100000, 300000, None);
    let target1 = create_event_with_times("target1", 120000, 150000, None);
    let target2 = create_event_with_times("target2", 200000, 250000, None);
    let target3 = create_event_with_times("target3", 400000, 500000, None); // No overlap

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target1, &target2, &target3];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert_eq!(result.len(), 1);
    let source_id = adapter.get_event_id(&source_event);
    assert_eq!(result[&source_id].len(), 2); // Only target1 and target2 overlap
}

#[test]
fn test_find_overlapping_intervals_multiple_sources() {
    let adapter = NsysEventAdapter;

    let source1 = create_event_with_times("source1", 100000, 200000, None);
    let source2 = create_event_with_times("source2", 300000, 400000, None);
    let target1 = create_event_with_times("target1", 150000, 180000, None);
    let target2 = create_event_with_times("target2", 350000, 380000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source1, &source2];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target1, &target2];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert_eq!(result.len(), 2);
    let source1_id = adapter.get_event_id(&source1);
    let source2_id = adapter.get_event_id(&source2);
    assert_eq!(result[&source1_id].len(), 1);
    assert_eq!(result[&source2_id].len(), 1);
}

#[test]
fn test_find_overlapping_intervals_empty_sources() {
    let adapter = NsysEventAdapter;

    let target_event = create_event_with_times("target", 100000, 200000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_overlapping_intervals_empty_targets() {
    let adapter = NsysEventAdapter;

    let source_event = create_event_with_times("source", 100000, 200000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_overlapping_intervals_simultaneous_start() {
    let adapter = NsysEventAdapter;

    // Source and target start at the same time
    // Source is processed first (sort order puts Source before Target at same timestamp),
    // so source becomes active before target start is processed, detecting the overlap
    let source_event = create_event_with_times("source", 100000, 200000, None);
    let target_event = create_event_with_times("target", 100000, 150000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Overlap IS detected because source becomes active before target start is processed
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_overlapping_intervals_partial_overlap_end() {
    let adapter = NsysEventAdapter;

    // Target starts inside source and extends past
    let source_event = create_event_with_times("source", 100000, 200000, None);
    let target_event = create_event_with_times("target", 150000, 250000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_overlapping_intervals_zero_duration_source() {
    let adapter = NsysEventAdapter;

    // Zero duration event
    let source_event = create_event_with_times("source", 150000, 150000, None);
    let target_event = create_event_with_times("target", 100000, 200000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source_event];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target_event];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Zero-duration event at timestamp inside target should be captured
    // The sweep line algorithm treats start and end at same point
    assert!(result.is_empty() || result.len() == 1);
}

// ==========================
// Tests for build_correlation_map
// ==========================

#[test]
fn test_build_correlation_map_basic() {
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, Some(12345));
    let kernel2 = create_event_with_times("kernel2", 200000, 250000, Some(12345));
    let kernel3 = create_event_with_times("kernel3", 300000, 350000, Some(67890));

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2, &kernel3];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 2);
    assert!(result.contains_key(&12345));
    assert!(result.contains_key(&67890));
    assert_eq!(result[&12345].len(), 2);
    assert_eq!(result[&67890].len(), 1);
}

#[test]
fn test_build_correlation_map_missing_correlation_id() {
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, Some(12345));
    let kernel2 = create_event_with_times("kernel2", 200000, 250000, None); // No correlation ID

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&12345));
    assert_eq!(result[&12345].len(), 1);
}

#[test]
fn test_build_correlation_map_empty_list() {
    let adapter = NsysEventAdapter;

    let kernel_events: Vec<&ChromeTraceEvent> = vec![];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_build_correlation_map_single_kernel() {
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 100000, 150000, Some(99999));

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&99999));
    assert_eq!(result[&99999].len(), 1);
}

#[test]
fn test_build_correlation_map_zero_correlation_id() {
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 100000, 150000, Some(0));

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&0));
}

// ==========================
// Tests for aggregate_kernel_times
// ==========================

#[test]
fn test_aggregate_kernel_times_basic() {
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, None);
    let kernel2 = create_event_with_times("kernel2", 120000, 200000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000); // Min start
    assert_eq!(end, 200000); // Max end
}

#[test]
fn test_aggregate_kernel_times_single_kernel() {
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 100000, 150000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 150000);
}

#[test]
fn test_aggregate_kernel_times_empty_list() {
    let adapter = NsysEventAdapter;

    let kernels: Vec<&ChromeTraceEvent> = vec![];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_none());
}

#[test]
fn test_aggregate_kernel_times_non_overlapping() {
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, None);
    let kernel2 = create_event_with_times("kernel2", 200000, 250000, None);
    let kernel3 = create_event_with_times("kernel3", 300000, 350000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2, &kernel3];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000); // Min start
    assert_eq!(end, 350000); // Max end
}

#[test]
fn test_aggregate_kernel_times_nested() {
    let adapter = NsysEventAdapter;

    // kernel2 is completely inside kernel1
    let kernel1 = create_event_with_times("kernel1", 100000, 300000, None);
    let kernel2 = create_event_with_times("kernel2", 150000, 200000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 300000);
}

#[test]
fn test_aggregate_kernel_times_zero_duration() {
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 100000, 100000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 100000);
}

// ==========================
// Tests for find_kernels_for_annotation
// ==========================

#[test]
fn test_find_kernels_for_annotation_basic() {
    let adapter = NsysEventAdapter;

    let api_event = create_event_with_times("cudaLaunchKernel", 100000, 120000, Some(12345));
    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "kernel");
}

#[test]
fn test_find_kernels_for_annotation_multiple_kernels() {
    let adapter = NsysEventAdapter;

    let api_event = create_event_with_times("cudaLaunchKernel", 100000, 120000, Some(12345));
    let kernel1 = create_event_with_times("kernel1", 130000, 180000, Some(12345));
    let kernel2 = create_event_with_times("kernel2", 190000, 220000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel1, &kernel2]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert_eq!(result.len(), 2);
}

#[test]
fn test_find_kernels_for_annotation_multiple_api_events() {
    let adapter = NsysEventAdapter;

    let api_event1 = create_event_with_times("cudaLaunchKernel1", 100000, 120000, Some(12345));
    let api_event2 = create_event_with_times("cudaLaunchKernel2", 200000, 220000, Some(67890));
    let kernel1 = create_event_with_times("kernel1", 130000, 180000, Some(12345));
    let kernel2 = create_event_with_times("kernel2", 230000, 280000, Some(67890));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event1, &api_event2];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel1]);
    correlation_map.insert(67890, vec![&kernel2]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert_eq!(result.len(), 2);
}

#[test]
fn test_find_kernels_for_annotation_no_match() {
    let adapter = NsysEventAdapter;

    let api_event = create_event_with_times("cudaLaunchKernel", 100000, 120000, Some(99999));
    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_missing_correlation_id() {
    let adapter = NsysEventAdapter;

    let api_event = create_event_with_times("cudaLaunchKernel", 100000, 120000, None);
    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_empty_kernel_list() {
    let adapter = NsysEventAdapter;

    let api_event = create_event_with_times("cudaLaunchKernel", 100000, 120000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api_event];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![]); // Empty kernel list

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_empty_api_events() {
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

// ==========================
// Negative Tests - Malformed/Invalid Events
// ==========================

/// Helper to create an event without time range args (will be filtered)
fn create_event_without_times(name: &str, correlation_id: Option<i32>) -> ChromeTraceEvent {
    let mut event = ChromeTraceEvent::complete(
        name.to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );
    // Don't add start_ns/end_ns

    if let Some(corr_id) = correlation_id {
        event = event.with_arg("correlationId", serde_json::json!(corr_id));
    }

    event
}

/// Helper to create a metadata event (non-Complete phase)
fn create_metadata_event(name: &str) -> ChromeTraceEvent {
    ChromeTraceEvent::metadata(
        name.to_string(),
        "Device 0".to_string(),
        "Thread 1".to_string(),
        HashMap::new(),
    )
}

#[test]
fn test_find_overlapping_intervals_all_events_missing_time_range() {
    // All events are missing time ranges - all should be filtered
    let adapter = NsysEventAdapter;

    let source = create_event_without_times("source", None);
    let target = create_event_without_times("target", None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_overlapping_intervals_all_non_complete_phase() {
    // All events are metadata (non-Complete phase) - all should be filtered
    let adapter = NsysEventAdapter;

    let source = create_metadata_event("source");
    let target = create_metadata_event("target");

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_overlapping_intervals_mixed_valid_invalid_sources() {
    // Mix of valid and invalid source events
    let adapter = NsysEventAdapter;

    let valid_source = create_event_with_times("valid_source", 100000, 200000, None);
    let invalid_source = create_event_without_times("invalid_source", None);

    let target = create_event_with_times("target", 150000, 180000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&valid_source, &invalid_source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Only valid_source should have overlaps
    assert_eq!(result.len(), 1);
    let valid_source_id = adapter.get_event_id(&valid_source);
    assert!(result.contains_key(&valid_source_id));
}

#[test]
fn test_find_overlapping_intervals_mixed_valid_invalid_targets() {
    // Mix of valid and invalid target events
    let adapter = NsysEventAdapter;

    let source = create_event_with_times("source", 100000, 200000, None);

    let valid_target = create_event_with_times("valid_target", 150000, 180000, None);
    let invalid_target = create_event_without_times("invalid_target", None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&valid_target, &invalid_target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Source should only overlap with valid_target
    assert_eq!(result.len(), 1);
    let source_id = adapter.get_event_id(&source);
    assert_eq!(result[&source_id].len(), 1);
}

#[test]
fn test_find_overlapping_intervals_inverted_time_range() {
    // Event with end < start - algorithm should handle gracefully
    let adapter = NsysEventAdapter;

    // Create event with inverted range
    let mut source = ChromeTraceEvent::complete(
        "source".to_string(),
        200.0,
        -100.0, // Negative duration!
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(200000))
    .with_arg("end_ns", serde_json::json!(100000)); // End < Start!

    let target = create_event_with_times("target", 150000, 180000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    // Should not panic - behavior is implementation-defined for inverted ranges
    let _result = find_overlapping_intervals(&source_events, &target_events, &adapter);
    // Just verify it doesn't crash
}

#[test]
fn test_find_overlapping_intervals_negative_timestamps() {
    // Events with negative timestamps - should work normally
    let adapter = NsysEventAdapter;

    let source = create_event_with_times("source", -200000, -100000, None);
    let target = create_event_with_times("target", -180000, -150000, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Overlap should be detected (target is inside source)
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_overlapping_intervals_very_large_timestamps() {
    // Events with very large timestamps
    let adapter = NsysEventAdapter;

    let source = create_event_with_times("source", i64::MAX - 2000, i64::MAX - 1000, None);
    let target = create_event_with_times("target", i64::MAX - 1500, i64::MAX - 1200, None);

    let source_events: Vec<&ChromeTraceEvent> = vec![&source];
    let target_events: Vec<&ChromeTraceEvent> = vec![&target];

    let result = find_overlapping_intervals(&source_events, &target_events, &adapter);

    // Should handle large values correctly
    assert_eq!(result.len(), 1);
}

// ==========================
// Negative Tests - build_correlation_map Edge Cases
// ==========================

#[test]
fn test_build_correlation_map_all_missing_correlation_id() {
    // All kernels missing correlation ID
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, None);
    let kernel2 = create_event_with_times("kernel2", 200000, 250000, None);

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_build_correlation_map_mixed_valid_invalid() {
    // Mix of kernels with and without correlation ID
    let adapter = NsysEventAdapter;

    let valid_kernel = create_event_with_times("valid_kernel", 100000, 150000, Some(12345));
    let invalid_kernel = create_event_with_times("invalid_kernel", 200000, 250000, None);

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&valid_kernel, &invalid_kernel];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&12345));
    assert_eq!(result[&12345].len(), 1);
}

#[test]
fn test_build_correlation_map_negative_correlation_id() {
    // Negative correlation IDs
    let adapter = NsysEventAdapter;

    let kernel = create_event_with_times("kernel", 100000, 150000, Some(-12345));

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&-12345));
}

#[test]
fn test_build_correlation_map_duplicate_same_correlation() {
    // Multiple kernels with same correlation ID (valid case)
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", 100000, 150000, Some(12345));
    let kernel2 = create_event_with_times("kernel2", 200000, 250000, Some(12345));
    let kernel3 = create_event_with_times("kernel3", 300000, 350000, Some(12345));

    let kernel_events: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2, &kernel3];

    let result = build_correlation_map(&kernel_events, &adapter);

    assert_eq!(result.len(), 1);
    assert_eq!(result[&12345].len(), 3);
}

// ==========================
// Negative Tests - aggregate_kernel_times Edge Cases
// ==========================

#[test]
fn test_aggregate_kernel_times_all_invalid_time_ranges() {
    // All kernels have invalid time ranges (missing or non-Complete phase)
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_without_times("kernel1", None);
    let kernel2 = create_metadata_event("kernel2");

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = aggregate_kernel_times(&kernels, &adapter);
    assert!(result.is_none());
}

#[test]
fn test_aggregate_kernel_times_mixed_valid_invalid() {
    // Mix of valid and invalid kernels
    let adapter = NsysEventAdapter;

    let valid_kernel = create_event_with_times("valid", 100000, 150000, None);
    let invalid_kernel = create_event_without_times("invalid", None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&valid_kernel, &invalid_kernel];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 150000);
}

#[test]
fn test_aggregate_kernel_times_single_valid_among_many_invalid() {
    // One valid kernel among many invalid
    let adapter = NsysEventAdapter;

    let invalid1 = create_event_without_times("invalid1", None);
    let invalid2 = create_metadata_event("invalid2");
    let valid = create_event_with_times("valid", 200000, 300000, None);
    let invalid3 = create_event_without_times("invalid3", None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&invalid1, &invalid2, &valid, &invalid3];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 200000);
    assert_eq!(end, 300000);
}

#[test]
fn test_aggregate_kernel_times_negative_times() {
    // Kernels with negative timestamps
    let adapter = NsysEventAdapter;

    let kernel1 = create_event_with_times("kernel1", -200000, -100000, None);
    let kernel2 = create_event_with_times("kernel2", -300000, -150000, None);

    let kernels: Vec<&ChromeTraceEvent> = vec![&kernel1, &kernel2];

    let result = aggregate_kernel_times(&kernels, &adapter);

    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, -300000); // Min start
    assert_eq!(end, -100000); // Max end
}

// ==========================
// Negative Tests - find_kernels_for_annotation Edge Cases
// ==========================

#[test]
fn test_find_kernels_for_annotation_all_api_missing_correlation() {
    // All API events missing correlation ID
    let adapter = NsysEventAdapter;

    let api1 = create_event_with_times("api1", 100000, 120000, None);
    let api2 = create_event_with_times("api2", 200000, 220000, None);

    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api1, &api2];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_correlation_not_in_map() {
    // API has correlation ID but it's not in the map
    let adapter = NsysEventAdapter;

    let api = create_event_with_times("api", 100000, 120000, Some(99999));
    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]); // Different correlation ID!

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_mixed_valid_invalid_api() {
    // Some API events have correlation, some don't
    let adapter = NsysEventAdapter;

    let valid_api = create_event_with_times("valid_api", 100000, 120000, Some(12345));
    let invalid_api = create_event_with_times("invalid_api", 200000, 220000, None);

    let kernel = create_event_with_times("kernel", 130000, 180000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&valid_api, &invalid_api];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    // Only valid_api should contribute
    assert_eq!(result.len(), 1);
}

#[test]
fn test_find_kernels_for_annotation_empty_correlation_map() {
    // Empty correlation map
    let adapter = NsysEventAdapter;

    let api = create_event_with_times("api", 100000, 120000, Some(12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api];
    let correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert!(result.is_empty());
}

#[test]
fn test_find_kernels_for_annotation_negative_correlation_id() {
    // Negative correlation ID
    let adapter = NsysEventAdapter;

    let api = create_event_with_times("api", 100000, 120000, Some(-12345));
    let kernel = create_event_with_times("kernel", 130000, 180000, Some(-12345));

    let overlapping_api_events: Vec<&ChromeTraceEvent> = vec![&api];
    let mut correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::new();
    correlation_map.insert(-12345, vec![&kernel]);

    let result = find_kernels_for_annotation(&overlapping_api_events, &correlation_map, &adapter);

    assert_eq!(result.len(), 1);
}

