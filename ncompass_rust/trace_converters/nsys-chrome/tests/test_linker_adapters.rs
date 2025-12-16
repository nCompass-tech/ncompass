//! Integration tests for linker adapters module

use nsys_chrome::linker::adapters::{EventAdapter, NsysEventAdapter};
use nsys_chrome::models::ChromeTraceEvent;
use std::collections::HashMap;

// ==========================
// Tests for NsysEventAdapter
// ==========================

#[test]
fn test_get_time_range_valid() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000))
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 150000);
}

#[test]
fn test_get_time_range_missing_start_ns() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_missing_end_ns() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_zero_duration() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        0.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000))
    .with_arg("end_ns", serde_json::json!(100000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 100000);
    assert_eq!(end, 100000);
}

#[test]
fn test_get_correlation_id_valid() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(12345));

    let result = adapter.get_correlation_id(&event);
    assert_eq!(result, Some(12345));
}

#[test]
fn test_get_correlation_id_missing() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_zero() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(0));

    let result = adapter.get_correlation_id(&event);
    assert_eq!(result, Some(0));
}

#[test]
fn test_get_event_id_unique_per_instance() {
    let adapter = NsysEventAdapter;
    let event1 = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );
    let event2 = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let id1 = adapter.get_event_id(&event1);
    let id2 = adapter.get_event_id(&event2);

    // Different instances should have different IDs
    assert_ne!(id1, id2);
}

#[test]
fn test_get_event_id_stable_for_same_reference() {
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let id1 = adapter.get_event_id(&event);
    let id2 = adapter.get_event_id(&event);

    // Same reference should have same ID
    assert_eq!(id1, id2);
}

// ==========================
// Negative Tests - Malformed Data Types
// ==========================

#[test]
fn test_get_time_range_wrong_type_string_start() {
    // start_ns as string instead of i64 - should gracefully return None
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!("100000")) // String instead of i64!
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_wrong_type_string_end() {
    // end_ns as string instead of i64
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000))
    .with_arg("end_ns", serde_json::json!("150000")); // String instead of i64!

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_null_start() {
    // Explicit null value for start_ns
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::Value::Null)
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_null_end() {
    // Explicit null value for end_ns
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000))
    .with_arg("end_ns", serde_json::Value::Null);

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_float_timestamps() {
    // Float values for timestamps - as_i64() returns None for floats with decimals
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(100000.5)) // Float with decimal!
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    // Floats with non-zero decimal parts are NOT valid i64
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_array_value() {
    // Array instead of scalar
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!([100000, 200000])) // Array!
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_object_value() {
    // Object instead of scalar
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!({"value": 100000})) // Object!
    .with_arg("end_ns", serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

// ==========================
// Negative Tests - Boundary Values
// ==========================

#[test]
fn test_get_time_range_negative_timestamps() {
    // Negative timestamps - valid in some contexts, should work
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(-100000))
    .with_arg("end_ns", serde_json::json!(-50000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, -100000);
    assert_eq!(end, -50000);
}

#[test]
fn test_get_time_range_inverted_range() {
    // End before start - adapter should NOT validate, just return values
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(200000))
    .with_arg("end_ns", serde_json::json!(100000)); // End < Start!

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, 200000);
    assert_eq!(end, 100000); // Returns as-is, no validation
}

#[test]
fn test_get_time_range_max_i64_values() {
    // i64::MAX boundary
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(i64::MAX - 1000))
    .with_arg("end_ns", serde_json::json!(i64::MAX));

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, i64::MAX - 1000);
    assert_eq!(end, i64::MAX);
}

#[test]
fn test_get_time_range_min_i64_values() {
    // i64::MIN boundary
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(i64::MIN))
    .with_arg("end_ns", serde_json::json!(i64::MIN + 1000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_some());
    let (start, end) = result.unwrap();
    assert_eq!(start, i64::MIN);
    assert_eq!(end, i64::MIN + 1000);
}

// ==========================
// Negative Tests - Non-Complete Phase Events
// ==========================

#[test]
fn test_get_time_range_metadata_phase() {
    // Metadata event (phase "M") should return None even with time args
    let adapter = NsysEventAdapter;
    let mut args = HashMap::new();
    args.insert("name".to_string(), serde_json::json!("test_process"));

    let mut event = ChromeTraceEvent::metadata(
        "process_name".to_string(),
        "Device 0".to_string(),
        "Thread 1".to_string(),
        args,
    );
    // Add time args even though it's a metadata event
    event
        .args
        .insert("start_ns".to_string(), serde_json::json!(100000));
    event
        .args
        .insert("end_ns".to_string(), serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    // Metadata phase should return None regardless of time args
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_flow_start_phase() {
    // Flow start event should return None
    let adapter = NsysEventAdapter;
    let mut event = ChromeTraceEvent::flow_start(
        100.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        nsys_chrome::models::StringOrInt::Int(12345),
    );
    event
        .args
        .insert("start_ns".to_string(), serde_json::json!(100000));
    event
        .args
        .insert("end_ns".to_string(), serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_time_range_flow_finish_phase() {
    // Flow finish event should return None
    let adapter = NsysEventAdapter;
    let mut event = ChromeTraceEvent::flow_finish(
        100.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        nsys_chrome::models::StringOrInt::Int(12345),
        nsys_chrome::models::BindingPoint::Enclosing,
    );
    event
        .args
        .insert("start_ns".to_string(), serde_json::json!(100000));
    event
        .args
        .insert("end_ns".to_string(), serde_json::json!(150000));

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

// ==========================
// Negative Tests - Correlation ID Edge Cases
// ==========================

#[test]
fn test_get_correlation_id_wrong_type_string() {
    // correlationId as string instead of number
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!("12345")); // String!

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_null() {
    // correlationId as explicit null
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::Value::Null);

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_negative() {
    // Negative correlation IDs - unusual but should work
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(-12345));

    let result = adapter.get_correlation_id(&event);
    assert_eq!(result, Some(-12345));
}

#[test]
fn test_get_correlation_id_float() {
    // correlationId as float - should fail
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(12345.5)); // Float!

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_array() {
    // correlationId as array - should fail
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!([12345])); // Array!

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_i32_max() {
    // i32::MAX boundary
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(i32::MAX as i64));

    let result = adapter.get_correlation_id(&event);
    assert_eq!(result, Some(i32::MAX));
}

#[test]
fn test_get_correlation_id_i32_min() {
    // i32::MIN boundary
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_arg("correlationId", serde_json::json!(i32::MIN as i64));

    let result = adapter.get_correlation_id(&event);
    assert_eq!(result, Some(i32::MIN));
}

// ==========================
// Negative Tests - Empty/Malformed Args
// ==========================

#[test]
fn test_get_time_range_empty_args() {
    // Event with no args at all
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );
    // Don't add any args

    let result = adapter.get_time_range(&event);
    assert!(result.is_none());
}

#[test]
fn test_get_correlation_id_empty_args() {
    // Event with no args at all
    let adapter = NsysEventAdapter;
    let event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let result = adapter.get_correlation_id(&event);
    assert!(result.is_none());
}

