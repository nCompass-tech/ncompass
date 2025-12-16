//! Unit tests for NVTX linker module

use nsys_chrome::linker::link_nvtx_to_kernels;
use nsys_chrome::models::{ChromeTraceEvent, ConversionOptions};
use std::collections::HashMap;

// ==========================
// Helper Functions
// ==========================

/// Create an NVTX event with required fields for linking
fn create_nvtx_event(
    name: &str,
    start_ns: i64,
    end_ns: i64,
    device_id: i32,
    tid: i32,
) -> ChromeTraceEvent {
    ChromeTraceEvent::complete(
        name.to_string(),
        start_ns as f64 / 1000.0,
        (end_ns - start_ns) as f64 / 1000.0,
        format!("Device {}", device_id),
        format!("NVTX Thread {}", tid),
        "nvtx".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(start_ns))
    .with_arg("end_ns", serde_json::json!(end_ns))
    .with_arg("deviceId", serde_json::json!(device_id))
    .with_arg("raw_tid", serde_json::json!(tid))
}

/// Create a CUDA API event with required fields for linking
fn create_cuda_api_event(
    name: &str,
    start_ns: i64,
    end_ns: i64,
    device_id: i32,
    tid: i32,
    correlation_id: i32,
) -> ChromeTraceEvent {
    ChromeTraceEvent::complete(
        name.to_string(),
        start_ns as f64 / 1000.0,
        (end_ns - start_ns) as f64 / 1000.0,
        format!("Device {}", device_id),
        format!("CUDA API Thread {}", tid),
        "cuda_api".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(start_ns))
    .with_arg("end_ns", serde_json::json!(end_ns))
    .with_arg("deviceId", serde_json::json!(device_id))
    .with_arg("raw_tid", serde_json::json!(tid))
    .with_arg("correlationId", serde_json::json!(correlation_id))
}

/// Create a kernel event with required fields for linking
fn create_kernel_event(
    name: &str,
    start_ns: i64,
    end_ns: i64,
    device_id: i32,
    stream_id: i32,
    correlation_id: i32,
) -> ChromeTraceEvent {
    ChromeTraceEvent::complete(
        name.to_string(),
        start_ns as f64 / 1000.0,
        (end_ns - start_ns) as f64 / 1000.0,
        format!("Device {}", device_id),
        format!("Stream {}", stream_id),
        "kernel".to_string(),
    )
    .with_arg("start_ns", serde_json::json!(start_ns))
    .with_arg("end_ns", serde_json::json!(end_ns))
    .with_arg("deviceId", serde_json::json!(device_id))
    .with_arg("streamId", serde_json::json!(stream_id))
    .with_arg("correlationId", serde_json::json!(correlation_id))
}

// ==========================
// Tests for link_nvtx_to_kernels
// ==========================

#[test]
fn test_link_nvtx_to_kernels_basic() {
    // Create a basic scenario: NVTX event overlaps with CUDA API which correlates to kernel
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("matmul_kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should create one nvtx-kernel event
    assert_eq!(nvtx_kernel_events.len(), 1);
    assert_eq!(nvtx_kernel_events[0].name, "forward");
    assert_eq!(nvtx_kernel_events[0].cat, "nvtx-kernel");

    // Should have one mapped identifier
    assert_eq!(mapped_identifiers.len(), 1);

    // Should create flow events (start and finish for the link)
    assert!(!flow_events.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_no_overlap() {
    // NVTX event doesn't overlap with CUDA API
    let nvtx_event = create_nvtx_event("forward", 100000, 150000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 200000, 220000, 0, 1, 12345);
    let kernel_event = create_kernel_event("matmul_kernel", 230000, 280000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No nvtx-kernel events should be created (no overlap)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
    // But flow events for cuda_api -> kernel should still exist
    assert!(!flow_events.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_multiple_kernels() {
    // One NVTX event covering multiple kernel launches
    let nvtx_event = create_nvtx_event("forward", 100000, 300000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel1 = create_kernel_event("kernel1", 140000, 180000, 0, 1, 12345);
    let kernel2 = create_kernel_event("kernel2", 190000, 230000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel1, kernel2];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should create one nvtx-kernel event spanning both kernels
    assert_eq!(nvtx_kernel_events.len(), 1);
    assert_eq!(nvtx_kernel_events[0].name, "forward");

    // Duration should span from first kernel start (140000) to last kernel end (230000)
    // In microseconds: 140.0 to 230.0
    assert_eq!(nvtx_kernel_events[0].ts, 140.0);
    assert_eq!(nvtx_kernel_events[0].dur.unwrap(), 90.0); // 230 - 140

    assert_eq!(mapped_identifiers.len(), 1);
}

#[test]
fn test_link_nvtx_to_kernels_multiple_cuda_api_calls() {
    // One NVTX event covering multiple CUDA API calls
    let nvtx_event = create_nvtx_event("forward", 100000, 400000, 0, 1);
    let cuda_api1 = create_cuda_api_event("cudaLaunchKernel1", 110000, 130000, 0, 1, 12345);
    let cuda_api2 = create_cuda_api_event("cudaLaunchKernel2", 200000, 220000, 0, 1, 67890);
    let kernel1 = create_kernel_event("kernel1", 140000, 180000, 0, 1, 12345);
    let kernel2 = create_kernel_event("kernel2", 230000, 280000, 0, 1, 67890);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api1, cuda_api2];
    let kernel_events = vec![kernel1, kernel2];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should create one nvtx-kernel event spanning all kernels
    assert_eq!(nvtx_kernel_events.len(), 1);
    // Duration should span from kernel1 start (140000) to kernel2 end (280000)
    assert_eq!(nvtx_kernel_events[0].ts, 140.0);
    assert_eq!(nvtx_kernel_events[0].dur.unwrap(), 140.0); // 280 - 140

    assert_eq!(mapped_identifiers.len(), 1);
}

#[test]
fn test_link_nvtx_to_kernels_multiple_nvtx_events() {
    // Multiple NVTX events, each overlapping with different CUDA API calls
    let nvtx1 = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let nvtx2 = create_nvtx_event("backward", 300000, 400000, 0, 1);

    let cuda_api1 = create_cuda_api_event("cudaLaunchKernel1", 110000, 130000, 0, 1, 12345);
    let cuda_api2 = create_cuda_api_event("cudaLaunchKernel2", 310000, 330000, 0, 1, 67890);

    let kernel1 = create_kernel_event("kernel1", 140000, 180000, 0, 1, 12345);
    let kernel2 = create_kernel_event("kernel2", 340000, 380000, 0, 1, 67890);

    let nvtx_events = vec![nvtx1, nvtx2];
    let cuda_api_events = vec![cuda_api1, cuda_api2];
    let kernel_events = vec![kernel1, kernel2];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should create two nvtx-kernel events
    assert_eq!(nvtx_kernel_events.len(), 2);
    assert_eq!(mapped_identifiers.len(), 2);
}

#[test]
fn test_link_nvtx_to_kernels_multiple_devices() {
    // Events on different devices
    let nvtx1 = create_nvtx_event("forward_dev0", 100000, 200000, 0, 1);
    let nvtx2 = create_nvtx_event("forward_dev1", 100000, 200000, 1, 1);

    let cuda_api1 = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let cuda_api2 = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 1, 1, 67890);

    let kernel1 = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);
    let kernel2 = create_kernel_event("kernel", 140000, 180000, 1, 1, 67890);

    let nvtx_events = vec![nvtx1, nvtx2];
    let cuda_api_events = vec![cuda_api1, cuda_api2];
    let kernel_events = vec![kernel1, kernel2];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should create two nvtx-kernel events (one per device)
    assert_eq!(nvtx_kernel_events.len(), 2);
    assert_eq!(mapped_identifiers.len(), 2);
}

#[test]
fn test_link_nvtx_to_kernels_no_common_devices() {
    // NVTX on device 0, CUDA API and kernel on device 1
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 1, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 1, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking should happen (no common device)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_empty_inputs() {
    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&[], &[], &[], &options);

    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
    assert!(flow_events.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_missing_time_fields() {
    // NVTX event without start_ns/end_ns should be filtered
    let mut nvtx_event = ChromeTraceEvent::complete(
        "forward".to_string(),
        100.0,
        100.0,
        "Device 0".to_string(),
        "NVTX Thread 1".to_string(),
        "nvtx".to_string(),
    );
    nvtx_event.args.insert("deviceId".to_string(), serde_json::json!(0));
    // Missing start_ns and end_ns

    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // NVTX without time fields should be filtered, no linking
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_with_color_scheme() {
    // Test color scheme application
    let nvtx_event = create_nvtx_event("compute_forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let mut color_scheme = HashMap::new();
    color_scheme.insert("compute.*".to_string(), "thread_state_running".to_string());

    let options = ConversionOptions {
        activity_types: vec![
            "kernel".to_string(),
            "nvtx".to_string(),
            "nvtx-kernel".to_string(),
        ],
        nvtx_event_prefix: None,
        nvtx_color_scheme: color_scheme,
        include_metadata: true,
    };

    let (nvtx_kernel_events, _mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert_eq!(nvtx_kernel_events.len(), 1);
    // Color should be applied based on regex match
    assert_eq!(
        nvtx_kernel_events[0].cname,
        Some("thread_state_running".to_string())
    );
}

#[test]
fn test_link_nvtx_to_kernels_color_scheme_no_match() {
    // Test color scheme when pattern doesn't match
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let mut color_scheme = HashMap::new();
    color_scheme.insert("compute.*".to_string(), "thread_state_running".to_string());

    let options = ConversionOptions {
        activity_types: vec![
            "kernel".to_string(),
            "nvtx".to_string(),
            "nvtx-kernel".to_string(),
        ],
        nvtx_event_prefix: None,
        nvtx_color_scheme: color_scheme,
        include_metadata: true,
    };

    let (nvtx_kernel_events, _mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert_eq!(nvtx_kernel_events.len(), 1);
    // Color should not be applied (pattern doesn't match)
    assert!(nvtx_kernel_events[0].cname.is_none());
}

#[test]
fn test_link_nvtx_to_kernels_flow_events_structure() {
    // Verify flow events have correct structure
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (_nvtx_kernel_events, _mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should have flow start and flow finish events
    assert_eq!(flow_events.len(), 2);

    // Find flow start and finish
    let flow_start = flow_events
        .iter()
        .find(|e| e.ph == nsys_chrome::models::ChromeTracePhase::FlowStart);
    let flow_finish = flow_events
        .iter()
        .find(|e| e.ph == nsys_chrome::models::ChromeTracePhase::FlowFinish);

    assert!(flow_start.is_some());
    assert!(flow_finish.is_some());

    // Verify flow event properties
    let flow_start = flow_start.unwrap();
    let flow_finish = flow_finish.unwrap();

    assert_eq!(flow_start.cat, "cuda_flow");
    assert_eq!(flow_finish.cat, "cuda_flow");
    assert!(flow_start.id.is_some());
    assert!(flow_finish.id.is_some());
    assert!(flow_finish.bp.is_some());
}

#[test]
fn test_link_nvtx_to_kernels_cuda_api_no_correlation() {
    // CUDA API event without correlation ID
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);

    let mut cuda_api_event = ChromeTraceEvent::complete(
        "cudaLaunchKernel".to_string(),
        110.0,
        20.0,
        "Device 0".to_string(),
        "CUDA API Thread 1".to_string(),
        "cuda_api".to_string(),
    );
    cuda_api_event
        .args
        .insert("start_ns".to_string(), serde_json::json!(110000));
    cuda_api_event
        .args
        .insert("end_ns".to_string(), serde_json::json!(130000));
    cuda_api_event
        .args
        .insert("deviceId".to_string(), serde_json::json!(0));
    // Missing correlationId

    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking should happen (CUDA API has no correlation ID)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_kernel_time_aggregation() {
    // Verify that kernel times are correctly aggregated (min start, max end)
    let nvtx_event = create_nvtx_event("forward", 100000, 400000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);

    // Kernels with different start/end times
    let kernel1 = create_kernel_event("kernel1", 150000, 200000, 0, 1, 12345); // Later start
    let kernel2 = create_kernel_event("kernel2", 140000, 250000, 0, 1, 12345); // Earlier start, later end

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel1, kernel2];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, _mapped_identifiers, _flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert_eq!(nvtx_kernel_events.len(), 1);
    // Should use min start (140000) and max end (250000)
    assert_eq!(nvtx_kernel_events[0].ts, 140.0);
    assert_eq!(nvtx_kernel_events[0].dur.unwrap(), 110.0); // 250 - 140
}

// ==========================
// Negative Tests - Invalid Regex in Color Scheme
// ==========================

#[test]
fn test_link_nvtx_to_kernels_invalid_regex_pattern() {
    // Invalid regex pattern - should not crash, just skip applying color
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let mut color_scheme = HashMap::new();
    color_scheme.insert("[invalid(regex".to_string(), "thread_state_running".to_string()); // Invalid regex!

    let options = ConversionOptions {
        activity_types: vec!["kernel".to_string(), "nvtx".to_string(), "nvtx-kernel".to_string()],
        nvtx_event_prefix: None,
        nvtx_color_scheme: color_scheme,
        include_metadata: true,
    };

    // Should not panic
    let (nvtx_kernel_events, _, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert_eq!(nvtx_kernel_events.len(), 1);
    // Color should not be applied (invalid regex skipped)
    assert!(nvtx_kernel_events[0].cname.is_none());
}

#[test]
fn test_link_nvtx_to_kernels_multiple_invalid_regex_patterns() {
    // Multiple invalid regex patterns
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let mut color_scheme = HashMap::new();
    color_scheme.insert("[bad(".to_string(), "color1".to_string());
    color_scheme.insert("*invalid*".to_string(), "color2".to_string()); // * without preceding char
    color_scheme.insert("(unclosed".to_string(), "color3".to_string());

    let options = ConversionOptions {
        activity_types: vec!["kernel".to_string(), "nvtx".to_string(), "nvtx-kernel".to_string()],
        nvtx_event_prefix: None,
        nvtx_color_scheme: color_scheme,
        include_metadata: true,
    };

    // Should not panic
    let (nvtx_kernel_events, _, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert_eq!(nvtx_kernel_events.len(), 1);
}

// ==========================
// Negative Tests - Wrong Data Types
// ==========================

#[test]
fn test_link_nvtx_to_kernels_device_id_as_string() {
    // deviceId as string instead of number - should be filtered
    let mut nvtx_event = ChromeTraceEvent::complete(
        "forward".to_string(),
        100.0,
        100.0,
        "Device 0".to_string(),
        "NVTX Thread 1".to_string(),
        "nvtx".to_string(),
    );
    nvtx_event.args.insert("deviceId".to_string(), serde_json::json!("0")); // String!
    nvtx_event.args.insert("start_ns".to_string(), serde_json::json!(100000));
    nvtx_event.args.insert("end_ns".to_string(), serde_json::json!(200000));
    nvtx_event.args.insert("raw_tid".to_string(), serde_json::json!(1));

    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // NVTX should be filtered (deviceId can't be parsed as i64)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_cuda_api_device_id_as_string() {
    // CUDA API deviceId as string
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);

    let mut cuda_api_event = ChromeTraceEvent::complete(
        "cudaLaunchKernel".to_string(),
        110.0,
        20.0,
        "Device 0".to_string(),
        "CUDA API Thread 1".to_string(),
        "cuda_api".to_string(),
    );
    cuda_api_event.args.insert("deviceId".to_string(), serde_json::json!("0")); // String!
    cuda_api_event.args.insert("start_ns".to_string(), serde_json::json!(110000));
    cuda_api_event.args.insert("end_ns".to_string(), serde_json::json!(130000));
    cuda_api_event.args.insert("correlationId".to_string(), serde_json::json!(12345));

    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking should happen (CUDA API filtered)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_kernel_device_id_as_string() {
    // Kernel deviceId as string
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);

    let mut kernel_event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        140.0,
        40.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );
    kernel_event.args.insert("deviceId".to_string(), serde_json::json!("0")); // String!
    kernel_event.args.insert("start_ns".to_string(), serde_json::json!(140000));
    kernel_event.args.insert("end_ns".to_string(), serde_json::json!(180000));
    kernel_event.args.insert("correlationId".to_string(), serde_json::json!(12345));

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking should happen (kernel filtered - no common device)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

// ==========================
// Negative Tests - Missing Required Fields
// ==========================

#[test]
fn test_link_nvtx_to_kernels_nvtx_missing_device_id() {
    // NVTX missing deviceId entirely
    let mut nvtx_event = ChromeTraceEvent::complete(
        "forward".to_string(),
        100.0,
        100.0,
        "Device 0".to_string(),
        "NVTX Thread 1".to_string(),
        "nvtx".to_string(),
    );
    // No deviceId!
    nvtx_event.args.insert("start_ns".to_string(), serde_json::json!(100000));
    nvtx_event.args.insert("end_ns".to_string(), serde_json::json!(200000));

    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // NVTX should be filtered (no deviceId)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_cuda_api_missing_device_id() {
    // CUDA API missing deviceId
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);

    let mut cuda_api_event = ChromeTraceEvent::complete(
        "cudaLaunchKernel".to_string(),
        110.0,
        20.0,
        "Device 0".to_string(),
        "CUDA API Thread 1".to_string(),
        "cuda_api".to_string(),
    );
    // No deviceId!
    cuda_api_event.args.insert("start_ns".to_string(), serde_json::json!(110000));
    cuda_api_event.args.insert("end_ns".to_string(), serde_json::json!(130000));
    cuda_api_event.args.insert("correlationId".to_string(), serde_json::json!(12345));

    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking (no common device with CUDA API)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_kernel_missing_device_id() {
    // Kernel missing deviceId
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);

    let mut kernel_event = ChromeTraceEvent::complete(
        "kernel".to_string(),
        140.0,
        40.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );
    // No deviceId!
    kernel_event.args.insert("start_ns".to_string(), serde_json::json!(140000));
    kernel_event.args.insert("end_ns".to_string(), serde_json::json!(180000));
    kernel_event.args.insert("correlationId".to_string(), serde_json::json!(12345));

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking (kernel filtered - no deviceId)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

// ==========================
// Negative Tests - Device Mismatches
// ==========================

#[test]
fn test_link_nvtx_to_kernels_nvtx_cuda_api_different_devices() {
    // NVTX on device 0, CUDA API on device 1 - no common device
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 1, 1, 12345); // Device 1!
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 1, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking (no common device with NVTX and CUDA API)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_nvtx_kernel_different_devices() {
    // NVTX and CUDA API on device 0, kernel on device 1
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 1, 1, 12345); // Device 1!

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No linking (kernel on different device)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_negative_device_id() {
    // Negative device IDs - should still work
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, -1, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, -1, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, -1, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Should work with negative device IDs
    assert_eq!(nvtx_kernel_events.len(), 1);
    assert_eq!(mapped_identifiers.len(), 1);
}

// ==========================
// Negative Tests - All Events Filtered
// ==========================

#[test]
fn test_link_nvtx_to_kernels_all_events_non_complete_phase() {
    // All events are metadata (non-Complete phase)
    let nvtx = ChromeTraceEvent::metadata(
        "nvtx".to_string(),
        "Device 0".to_string(),
        "Thread 1".to_string(),
        HashMap::new(),
    );
    let cuda_api = ChromeTraceEvent::metadata(
        "cuda_api".to_string(),
        "Device 0".to_string(),
        "Thread 1".to_string(),
        HashMap::new(),
    );
    let kernel = ChromeTraceEvent::metadata(
        "kernel".to_string(),
        "Device 0".to_string(),
        "Thread 1".to_string(),
        HashMap::new(),
    );

    let nvtx_events = vec![nvtx];
    let cuda_api_events = vec![cuda_api];
    let kernel_events = vec![kernel];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
    assert!(flow_events.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_all_events_missing_time_fields() {
    // All events missing start_ns/end_ns
    let mut nvtx = ChromeTraceEvent::complete(
        "nvtx".to_string(),
        100.0,
        100.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        "nvtx".to_string(),
    );
    nvtx.args.insert("deviceId".to_string(), serde_json::json!(0));
    // No start_ns/end_ns

    let mut cuda_api = ChromeTraceEvent::complete(
        "cuda_api".to_string(),
        110.0,
        20.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        "cuda_api".to_string(),
    );
    cuda_api.args.insert("deviceId".to_string(), serde_json::json!(0));
    cuda_api.args.insert("correlationId".to_string(), serde_json::json!(12345));
    // No start_ns/end_ns

    let mut kernel = ChromeTraceEvent::complete(
        "kernel".to_string(),
        140.0,
        40.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        "kernel".to_string(),
    );
    kernel.args.insert("deviceId".to_string(), serde_json::json!(0));
    kernel.args.insert("correlationId".to_string(), serde_json::json!(12345));
    // No start_ns/end_ns

    let nvtx_events = vec![nvtx];
    let cuda_api_events = vec![cuda_api];
    let kernel_events = vec![kernel];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Everything filtered - no time fields
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
}

// ==========================
// Negative Tests - Correlation Chain Broken
// ==========================

#[test]
fn test_link_nvtx_to_kernels_correlation_mismatch() {
    // CUDA API has different correlation ID than kernel
    let nvtx_event = create_nvtx_event("forward", 100000, 200000, 0, 1);
    let cuda_api_event = create_cuda_api_event("cudaLaunchKernel", 110000, 130000, 0, 1, 12345);
    let kernel_event = create_kernel_event("kernel", 140000, 180000, 0, 1, 99999); // Different!

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api_event];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, flow_events) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // No nvtx-kernel linking (correlation IDs don't match)
    assert!(nvtx_kernel_events.is_empty());
    assert!(mapped_identifiers.is_empty());
    // But flow events should still be empty (no matching pairs)
    assert!(flow_events.is_empty());
}

#[test]
fn test_link_nvtx_to_kernels_multiple_cuda_api_one_matches() {
    // Multiple CUDA API events, only one matches kernel
    let nvtx_event = create_nvtx_event("forward", 100000, 300000, 0, 1);
    let cuda_api1 = create_cuda_api_event("api1", 110000, 130000, 0, 1, 11111); // No matching kernel
    let cuda_api2 = create_cuda_api_event("api2", 150000, 170000, 0, 1, 12345); // Has matching kernel
    let cuda_api3 = create_cuda_api_event("api3", 200000, 220000, 0, 1, 33333); // No matching kernel

    let kernel_event = create_kernel_event("kernel", 180000, 220000, 0, 1, 12345);

    let nvtx_events = vec![nvtx_event];
    let cuda_api_events = vec![cuda_api1, cuda_api2, cuda_api3];
    let kernel_events = vec![kernel_event];

    let options = ConversionOptions::default();

    let (nvtx_kernel_events, mapped_identifiers, _) =
        link_nvtx_to_kernels(&nvtx_events, &cuda_api_events, &kernel_events, &options);

    // Only cuda_api2 -> kernel link should produce nvtx-kernel event
    assert_eq!(nvtx_kernel_events.len(), 1);
    assert_eq!(mapped_identifiers.len(), 1);
}

