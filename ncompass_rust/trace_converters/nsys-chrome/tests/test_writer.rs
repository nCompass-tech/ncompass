//! Unit tests for writer module

use flate2::read::GzDecoder;
use nsys_chrome::models::{ChromeTraceEvent, ChromeTracePhase};
use nsys_chrome::writer::{ChromeTraceWriter, OVERFLOW_PREFIX};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use tempfile::NamedTempFile;

// ==========================
// Tests for write
// ==========================

#[test]
fn test_write_chrome_trace_basic() {
    // Test that function writes valid JSON
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();
    let events = vec![];

    ChromeTraceWriter::write(output_path, events).unwrap();

    // Verify file exists and contains valid JSON
    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed["traceEvents"], serde_json::json!([]));
}

#[test]
fn test_write_chrome_trace_readable() {
    // Test that output can be read back and matches input
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        ChromeTraceEvent::complete(
            "event1".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "kernel".to_string(),
        ),
        ChromeTraceEvent::new(
            "event2".to_string(),
            ChromeTracePhase::DurationBegin,
            200.0,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "nvtx".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events.clone()).unwrap();

    // Read back and verify
    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["traceEvents"][0]["name"], "event1");
    assert_eq!(parsed["traceEvents"][1]["name"], "event2");
}

#[test]
fn test_write_chrome_trace_with_metadata() {
    // Test with realistic traceEvents structure including args and metadata
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let mut kernel_args = HashMap::new();
    kernel_args.insert("deviceId".to_string(), serde_json::json!(0));
    kernel_args.insert("streamId".to_string(), serde_json::json!(1));
    kernel_args.insert("gridDim".to_string(), serde_json::json!([256, 1, 1]));
    kernel_args.insert("blockDim".to_string(), serde_json::json!([128, 1, 1]));

    let mut metadata_args = HashMap::new();
    metadata_args.insert("name".to_string(), serde_json::json!("Device 0"));

    let mut nvtx_args = HashMap::new();
    nvtx_args.insert("color".to_string(), serde_json::json!("#FF0000"));

    let events = vec![
        ChromeTraceEvent::complete(
            "kernel_launch".to_string(),
            1000.5,
            250.75,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "cuda".to_string(),
        )
        .with_args(kernel_args),
        ChromeTraceEvent::metadata(
            "process_name".to_string(),
            "Device 0".to_string(),
            String::new(),
            metadata_args,
        ),
        ChromeTraceEvent::new(
            "nvtx_range".to_string(),
            ChromeTracePhase::DurationBegin,
            500.0,
            "Device 0".to_string(),
            "Thread 1".to_string(),
            "nvtx".to_string(),
        )
        .with_args(nvtx_args),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    // Verify content integrity
    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 3);

    // Check nested args are preserved
    let kernel_event = &parsed["traceEvents"][0];
    assert_eq!(
        kernel_event["args"]["gridDim"],
        serde_json::json!([256, 1, 1])
    );
    assert_eq!(
        kernel_event["args"]["blockDim"],
        serde_json::json!([128, 1, 1])
    );
}

#[test]
fn test_write_chrome_trace_empty_trace_events() {
    // Test with empty traceEvents list
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();
    let events = vec![];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"], serde_json::json!([]));
}

#[test]
fn test_write_chrome_trace_unicode_content() {
    // Test that unicode content is properly encoded
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let mut args = HashMap::new();
    args.insert(
        "description".to_string(),
        serde_json::json!("テスト説明"),
    );

    let events = vec![ChromeTraceEvent::complete(
        "test_事件_émoji_🚀".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        "test".to_string(),
    )
    .with_args(args)];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["name"], "test_事件_émoji_🚀");
    assert_eq!(
        parsed["traceEvents"][0]["args"]["description"],
        "テスト説明"
    );
}

// ==========================
// Tests for write_gz
// ==========================

#[test]
fn test_write_chrome_trace_gz_basic() {
    // Test that function writes valid gzip-compressed JSON
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();
    let events = vec![];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    // Verify file exists
    assert!(std::path::Path::new(output_path).exists());

    // Verify it's valid gzip by attempting to decompress
    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed["traceEvents"], serde_json::json!([]));
}

#[test]
fn test_write_chrome_trace_gz_readable() {
    // Test that output can be read back and matches input
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        ChromeTraceEvent::complete(
            "event1".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "kernel".to_string(),
        ),
        ChromeTraceEvent::new(
            "event2".to_string(),
            ChromeTracePhase::DurationBegin,
            200.0,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "nvtx".to_string(),
        ),
    ];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    // Read back and verify
    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["traceEvents"][0]["name"], "event1");
    assert_eq!(parsed["traceEvents"][1]["name"], "event2");
}

#[test]
fn test_write_chrome_trace_gz_with_trace_events() {
    // Test with realistic traceEvents structure including args and metadata
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let mut kernel_args = HashMap::new();
    kernel_args.insert("deviceId".to_string(), serde_json::json!(0));
    kernel_args.insert("streamId".to_string(), serde_json::json!(1));
    kernel_args.insert("gridDim".to_string(), serde_json::json!([256, 1, 1]));
    kernel_args.insert("blockDim".to_string(), serde_json::json!([128, 1, 1]));

    let mut metadata_args = HashMap::new();
    metadata_args.insert("name".to_string(), serde_json::json!("Device 0"));

    let mut nvtx_args = HashMap::new();
    nvtx_args.insert("color".to_string(), serde_json::json!("#FF0000"));

    let events = vec![
        ChromeTraceEvent::complete(
            "kernel_launch".to_string(),
            1000.5,
            250.75,
            "Device 0".to_string(),
            "Stream 1".to_string(),
            "cuda".to_string(),
        )
        .with_args(kernel_args),
        ChromeTraceEvent::metadata(
            "process_name".to_string(),
            "Device 0".to_string(),
            String::new(),
            metadata_args,
        ),
        ChromeTraceEvent::new(
            "nvtx_range".to_string(),
            ChromeTracePhase::DurationBegin,
            500.0,
            "Device 0".to_string(),
            "Thread 1".to_string(),
            "nvtx".to_string(),
        )
        .with_args(nvtx_args),
    ];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    // Verify file size is smaller than uncompressed would be (basic compression check)
    let file_size = std::fs::metadata(output_path).unwrap().len();
    // Note: with just a few events, compression might not reduce size significantly
    assert!(file_size > 0);

    // Verify content integrity
    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"].as_array().unwrap().len(), 3);

    // Check nested args are preserved
    let kernel_event = &parsed["traceEvents"][0];
    assert_eq!(
        kernel_event["args"]["gridDim"],
        serde_json::json!([256, 1, 1])
    );
    assert_eq!(
        kernel_event["args"]["blockDim"],
        serde_json::json!([128, 1, 1])
    );
}

#[test]
fn test_write_chrome_trace_gz_empty_trace_events() {
    // Test with empty traceEvents list
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();
    let events = vec![];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"], serde_json::json!([]));
}

#[test]
fn test_write_chrome_trace_gz_unicode_content() {
    // Test that unicode content is properly encoded
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let mut args = HashMap::new();
    args.insert(
        "description".to_string(),
        serde_json::json!("テスト説明"),
    );

    let events = vec![ChromeTraceEvent::complete(
        "test_事件_émoji_🚀".to_string(),
        100.0,
        50.0,
        "Device 0".to_string(),
        "Thread 1".to_string(),
        "test".to_string(),
    )
    .with_args(args)];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["name"], "test_事件_émoji_🚀");
    assert_eq!(
        parsed["traceEvents"][0]["args"]["description"],
        "テスト説明"
    );
}

// ==========================
// Tests for overlap handling
// ==========================

#[test]
fn test_overlap_no_overlap_sequential_events() {
    // Test that sequential non-overlapping events keep original tid
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=100, dur=50, ends at 150
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=160, dur=30, starts after A ends - no overlap
        ChromeTraceEvent::complete(
            "B".to_string(),
            160.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 7");
}

#[test]
fn test_overlap_exactly_adjacent() {
    // Test that events exactly adjacent (no gap, no overlap) keep original tid
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=100, dur=50, ends at 150
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=150, dur=30, starts exactly when A ends - no overlap
        ChromeTraceEvent::complete(
            "B".to_string(),
            150.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 7");
}

#[test]
fn test_overlap_fully_nested_keeps_original_tid() {
    // Test that fully nested events keep original tid (Perfetto allows this)
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=100, dur=100, ends at 200 (long event)
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            100.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=120, dur=30, ends at 150 - fully nested within A
        ChromeTraceEvent::complete(
            "B".to_string(),
            120.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    // Both should be on original track (nested is OK)
    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 7");
}

#[test]
fn test_overlap_partial_overlap_moves_to_overflow() {
    // Test that partial overlap moves event to overflow track
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=100, dur=50, ends at 150
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=140, dur=30, ends at 170 - starts before A ends, ends after A ends
        ChromeTraceEvent::complete(
            "B".to_string(),
            140.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(
        parsed["traceEvents"][1]["tid"],
        format!("{}Stream 7", OVERFLOW_PREFIX)
    );
}

#[test]
fn test_overlap_small_overlap_like_real_gpu_traces() {
    // Test partial overlap with very small overlap (like real GPU traces)
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=9659065, dur=976, ends at 9660041
        ChromeTraceEvent::complete(
            "device_kernel".to_string(),
            9659065.0,
            976.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=9660039, dur=33, ends at 9660072 - overlaps by ~2µs
        ChromeTraceEvent::complete(
            "device_kernel".to_string(),
            9660039.0,
            33.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(
        parsed["traceEvents"][1]["tid"],
        format!("{}Stream 7", OVERFLOW_PREFIX)
    );
}

#[test]
fn test_overlap_different_tracks_independent() {
    // Test that events on different tracks don't affect each other
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event on Stream 7
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            100.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event on Stream 8 at overlapping time - should NOT be affected
        ChromeTraceEvent::complete(
            "B".to_string(),
            120.0,
            50.0,
            "Device 0".to_string(),
            "Stream 8".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 8"); // Different track, unaffected
}

#[test]
fn test_overlap_different_pids_independent() {
    // Test that events on different pids don't affect each other
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event on Device 0
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            100.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event on Device 1 at overlapping time - should NOT be affected
        ChromeTraceEvent::complete(
            "B".to_string(),
            120.0,
            50.0,
            "Device 1".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 7"); // Different pid, unaffected
}

#[test]
fn test_overlap_non_complete_events_unchanged() {
    // Test that non-Complete phase events are not processed
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Complete event to establish overlap window
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            100.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // DurationBegin event at overlapping time - should NOT be moved
        ChromeTraceEvent::new(
            "B".to_string(),
            ChromeTracePhase::DurationBegin,
            120.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "nvtx".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(parsed["traceEvents"][1]["tid"], "Stream 7"); // Not a Complete event, unchanged
}

#[test]
fn test_overlap_track_reused_after_gap() {
    // Test that original track is reused after gap (no infinite tracks)
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        // Event A: ts=100, dur=50, ends at 150
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event B: ts=140, dur=30, ends at 170 - partial overlap, moves to overflow
        ChromeTraceEvent::complete(
            "B".to_string(),
            140.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        // Event C: ts=200, dur=20 - starts after both A and B end, should go back to original
        ChromeTraceEvent::complete(
            "C".to_string(),
            200.0,
            20.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write(output_path, events).unwrap();

    let content = std::fs::read_to_string(output_path).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(
        parsed["traceEvents"][1]["tid"],
        format!("{}Stream 7", OVERFLOW_PREFIX)
    );
    assert_eq!(parsed["traceEvents"][2]["tid"], "Stream 7"); // Back to original track
}

#[test]
fn test_overlap_gz_handles_partial_overlap() {
    // Test that write_gz also handles overlapping events
    let temp_file = NamedTempFile::new().unwrap();
    let output_path = temp_file.path().to_str().unwrap();

    let events = vec![
        ChromeTraceEvent::complete(
            "A".to_string(),
            100.0,
            50.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
        ChromeTraceEvent::complete(
            "B".to_string(),
            140.0,
            30.0,
            "Device 0".to_string(),
            "Stream 7".to_string(),
            "kernel".to_string(),
        ),
    ];

    ChromeTraceWriter::write_gz(output_path, events).unwrap();

    let file = File::open(output_path).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert_eq!(parsed["traceEvents"][0]["tid"], "Stream 7");
    assert_eq!(
        parsed["traceEvents"][1]["tid"],
        format!("{}Stream 7", OVERFLOW_PREFIX)
    );
}

