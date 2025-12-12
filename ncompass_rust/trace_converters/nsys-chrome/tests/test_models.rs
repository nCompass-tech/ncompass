//! Unit tests for models module

use nsys_chrome::models::{
    ns_to_us, BindingPoint, ChromeTraceEvent, ChromeTracePhase, ConversionOptions, StringOrInt,
};
use std::collections::HashMap;

// ==========================
// Tests for ns_to_us function
// ==========================

#[test]
fn test_ns_to_us_basic() {
    // Test basic nanosecond to microsecond conversion
    let result = ns_to_us(1_000_000);
    assert_eq!(result, 1000.0);
}

#[test]
fn test_ns_to_us_zero() {
    // Test conversion of zero nanoseconds
    let result = ns_to_us(0);
    assert_eq!(result, 0.0);
}

#[test]
fn test_ns_to_us_fractional() {
    // Test conversion with fractional microseconds
    let result = ns_to_us(1500);
    assert_eq!(result, 1.5);
}

#[test]
fn test_ns_to_us_large_value() {
    // Test conversion of large nanosecond value
    let result = ns_to_us(1_000_000_000);
    assert_eq!(result, 1_000_000.0);
}

#[test]
fn test_ns_to_us_small_value() {
    // Test conversion of small nanosecond value
    let result = ns_to_us(1);
    assert_eq!(result, 0.001);
}

// ==========================
// Tests for ChromeTracePhase
// ==========================

#[test]
fn test_chrome_trace_phase_serialization() {
    use serde_json;

    // Test that phases serialize to correct strings
    assert_eq!(
        serde_json::to_string(&ChromeTracePhase::Complete).unwrap(),
        "\"X\""
    );
    assert_eq!(
        serde_json::to_string(&ChromeTracePhase::DurationBegin).unwrap(),
        "\"B\""
    );
    assert_eq!(
        serde_json::to_string(&ChromeTracePhase::DurationEnd).unwrap(),
        "\"E\""
    );
    assert_eq!(
        serde_json::to_string(&ChromeTracePhase::Instant).unwrap(),
        "\"i\""
    );
    assert_eq!(
        serde_json::to_string(&ChromeTracePhase::Metadata).unwrap(),
        "\"M\""
    );
}

// ==========================
// Tests for ChromeTraceEvent
// ==========================

#[test]
fn test_chrome_trace_event_new() {
    let event = ChromeTraceEvent::new(
        "test_event".to_string(),
        ChromeTracePhase::Complete,
        1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    assert_eq!(event.name, "test_event");
    assert_eq!(event.ph, ChromeTracePhase::Complete);
    assert_eq!(event.ts, 1000.0);
    assert_eq!(event.pid, "Device 0");
    assert_eq!(event.tid, "Stream 1");
    assert_eq!(event.cat, "kernel");
    assert!(event.args.is_empty());
    assert_eq!(event.dur, None);
}

#[test]
fn test_chrome_trace_event_complete() {
    let event = ChromeTraceEvent::complete(
        "kernel_launch".to_string(),
        1000.5,
        250.75,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "cuda".to_string(),
    );

    assert_eq!(event.name, "kernel_launch");
    assert_eq!(event.ph, ChromeTracePhase::Complete);
    assert_eq!(event.ts, 1000.5);
    assert_eq!(event.dur, Some(250.75));
    assert_eq!(event.pid, "Device 0");
    assert_eq!(event.tid, "Stream 1");
    assert_eq!(event.cat, "cuda");
}

#[test]
fn test_chrome_trace_event_metadata() {
    let mut args = HashMap::new();
    args.insert("name".to_string(), serde_json::json!("Device 0"));

    let event = ChromeTraceEvent::metadata(
        "process_name".to_string(),
        "Device 0".to_string(),
        String::new(),
        args,
    );

    assert_eq!(event.name, "process_name");
    assert_eq!(event.ph, ChromeTracePhase::Metadata);
    assert_eq!(event.ts, 0.0);
    assert_eq!(event.pid, "Device 0");
    assert_eq!(event.cat, "__metadata");
    assert_eq!(
        event.args.get("name").unwrap(),
        &serde_json::json!("Device 0")
    );
}

#[test]
fn test_chrome_trace_event_with_args() {
    let mut args = HashMap::new();
    args.insert("deviceId".to_string(), serde_json::json!(0));
    args.insert("streamId".to_string(), serde_json::json!(1));

    let event = ChromeTraceEvent::new(
        "test".to_string(),
        ChromeTracePhase::Complete,
        1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_args(args.clone());

    assert_eq!(event.args.get("deviceId").unwrap(), &serde_json::json!(0));
    assert_eq!(event.args.get("streamId").unwrap(), &serde_json::json!(1));
}

#[test]
fn test_chrome_trace_event_with_color() {
    let event = ChromeTraceEvent::new(
        "test".to_string(),
        ChromeTracePhase::Complete,
        1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    )
    .with_color("blue".to_string());

    assert_eq!(event.cname, Some("blue".to_string()));
}

#[test]
fn test_chrome_trace_event_serialization_complete() {
    use serde_json;

    let event = ChromeTraceEvent::complete(
        "test_kernel".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["name"], "test_kernel");
    assert_eq!(json["ph"], "X");
    assert_eq!(json["ts"], 100.0);
    assert_eq!(json["dur"], 50.0);
    assert_eq!(json["pid"], "Device 0");
    assert_eq!(json["tid"], "Stream 1");
    assert_eq!(json["cat"], "kernel");
}

#[test]
fn test_chrome_trace_event_serialization_skips_none() {
    use serde_json;

    let event = ChromeTraceEvent::new(
        "test".to_string(),
        ChromeTracePhase::DurationBegin,
        200.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let json_str = serde_json::to_string(&event).unwrap();
    // Should not contain optional fields that are None
    assert!(!json_str.contains("\"dur\""));
    assert!(!json_str.contains("\"cname\""));
    assert!(!json_str.contains("\"id\""));
    assert!(!json_str.contains("\"bp\""));
}

#[test]
fn test_chrome_trace_event_serialization_with_nested_args() {
    use serde_json;

    let mut args = HashMap::new();
    args.insert("deviceId".to_string(), serde_json::json!(0));
    args.insert("streamId".to_string(), serde_json::json!(1));
    args.insert("gridDim".to_string(), serde_json::json!([256, 1, 1]));
    args.insert("blockDim".to_string(), serde_json::json!([128, 1, 1]));

    let event = ChromeTraceEvent::complete(
        "kernel_launch".to_string(),
        1000.5,
        250.75,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "cuda".to_string(),
    )
    .with_args(args);

    let json = serde_json::to_value(&event).unwrap();
    assert_eq!(json["args"]["gridDim"], serde_json::json!([256, 1, 1]));
    assert_eq!(json["args"]["blockDim"], serde_json::json!([128, 1, 1]));
}

#[test]
fn test_chrome_trace_event_flow_start() {
    let event = ChromeTraceEvent::flow_start(
        1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        StringOrInt::Int(42),
    );

    assert_eq!(event.ph, ChromeTracePhase::FlowStart);
    assert_eq!(event.ts, 1000.0);
    assert_eq!(event.cat, "cuda_flow");
    assert_eq!(event.id, Some(StringOrInt::Int(42)));
}

#[test]
fn test_chrome_trace_event_flow_finish() {
    let event = ChromeTraceEvent::flow_finish(
        2000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        StringOrInt::Int(42),
        BindingPoint::Enclosing,
    );

    assert_eq!(event.ph, ChromeTracePhase::FlowFinish);
    assert_eq!(event.ts, 2000.0);
    assert_eq!(event.cat, "cuda_flow");
    assert_eq!(event.id, Some(StringOrInt::Int(42)));
    assert_eq!(event.bp, Some(BindingPoint::Enclosing));
}

// ==========================
// Tests for StringOrInt
// ==========================

#[test]
fn test_string_or_int_from_string() {
    let val = StringOrInt::from("test".to_string());
    match val {
        StringOrInt::String(s) => assert_eq!(s, "test"),
        _ => panic!("Expected String variant"),
    }
}

#[test]
fn test_string_or_int_from_i64() {
    let val = StringOrInt::from(42i64);
    match val {
        StringOrInt::Int(i) => assert_eq!(i, 42),
        _ => panic!("Expected Int variant"),
    }
}

#[test]
fn test_string_or_int_from_i32() {
    let val = StringOrInt::from(42i32);
    match val {
        StringOrInt::Int(i) => assert_eq!(i, 42),
        _ => panic!("Expected Int variant"),
    }
}

// ==========================
// Tests for ConversionOptions
// ==========================

#[test]
fn test_conversion_options_default() {
    let options = ConversionOptions::default();
    assert!(options.activity_types.contains(&"kernel".to_string()));
    assert!(options.activity_types.contains(&"nvtx".to_string()));
    assert!(options
        .activity_types
        .contains(&"nvtx-kernel".to_string()));
    assert!(options.activity_types.contains(&"cuda-api".to_string()));
    assert!(options.activity_types.contains(&"osrt".to_string()));
    assert!(options.activity_types.contains(&"sched".to_string()));
    assert_eq!(options.activity_types.len(), 6);
    assert_eq!(options.nvtx_event_prefix, None);
    assert!(options.nvtx_color_scheme.is_empty());
    assert!(options.include_metadata);
}

#[test]
fn test_conversion_options_custom() {
    let mut color_scheme = HashMap::new();
    color_scheme.insert("test_.*".to_string(), "blue".to_string());

    let options = ConversionOptions {
        activity_types: vec!["kernel".to_string(), "nvtx".to_string()],
        nvtx_event_prefix: Some(vec!["test_".to_string()]),
        nvtx_color_scheme: color_scheme.clone(),
        include_metadata: false,
    };

    assert_eq!(options.activity_types.len(), 2);
    assert!(options.activity_types.contains(&"kernel".to_string()));
    assert!(options.activity_types.contains(&"nvtx".to_string()));
    assert_eq!(
        options.nvtx_event_prefix,
        Some(vec!["test_".to_string()])
    );
    assert_eq!(
        options.nvtx_color_scheme.get("test_.*"),
        Some(&"blue".to_string())
    );
    assert!(!options.include_metadata);
}

