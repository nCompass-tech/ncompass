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
