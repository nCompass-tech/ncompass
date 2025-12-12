//! Integration tests for linker adapters module

use nsys_chrome::linker::adapters::{EventAdapter, NsysEventAdapter};
use nsys_chrome::models::ChromeTraceEvent;

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

