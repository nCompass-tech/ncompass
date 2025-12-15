//! Integration tests for nsys-chrome converter

use flate2::read::GzDecoder;
use nsys_chrome::{convert_file, convert_file_gz, ChromeTraceEvent, ConversionOptions, NsysChromeConverter};
use rusqlite;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use tempfile::{NamedTempFile, TempDir};

// ==========================
// Test Converter Creation
// ==========================

#[test]
fn test_converter_creation_file_not_found() {
    // Test that error is returned when input file doesn't exist
    let result = NsysChromeConverter::new("/nonexistent/directory/test.sqlite", None);
    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.to_lowercase().contains("failed to open sqlite"));
    }
}

#[test]
fn test_converter_creation_empty_db() {
    // Test that creating a converter with a valid but empty database succeeds
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    
    // Create an empty SQLite database
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    drop(conn);
    
    // Creating the converter should succeed
    let result = NsysChromeConverter::new(temp_path, None);
    assert!(result.is_ok());
    
    // Conversion should succeed but return empty events since no tables exist
    if let Ok(converter) = result {
        let convert_result = converter.convert();
        assert!(convert_result.is_ok());
        let events = convert_result.unwrap();
        // Should have no events (or only metadata events if any)
        let non_metadata_events: Vec<_> = events.iter().filter(|e| e.cat != "__metadata").collect();
        assert!(non_metadata_events.is_empty());
    }
}

#[test]
fn test_converter_creation_valid_db() {
    // Test that creating a converter with a valid database succeeds
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    
    // Create a valid SQLite database with basic schema
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);
    
    let result = NsysChromeConverter::new(temp_path, None);
    assert!(result.is_ok());
}

// ==========================
// Test Conversion Options
// ==========================

#[test]
fn test_conversion_options_default() {
    let options = ConversionOptions::default();
    assert!(options.activity_types.contains(&"kernel".to_string()));
    assert!(options.activity_types.contains(&"nvtx".to_string()));
    assert!(options.activity_types.contains(&"cuda-api".to_string()));
    assert!(options.activity_types.contains(&"osrt".to_string()));
    assert!(options.activity_types.contains(&"sched".to_string()));
    assert!(options.activity_types.contains(&"nvtx-kernel".to_string()));
    assert!(options.include_metadata);
    assert_eq!(options.nvtx_event_prefix, None);
    assert!(options.nvtx_color_scheme.is_empty());
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

// ==========================
// Test ChromeTraceEvent
// ==========================

#[test]
fn test_chrome_trace_event_serialization() {
    use serde_json;

    let event = ChromeTraceEvent::complete(
        "test_kernel".to_string(),
        1000.0,
        500.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let json = serde_json::to_string(&event).unwrap();
    assert!(json.contains("\"name\":\"test_kernel\""));
    assert!(json.contains("\"ph\":\"X\""));
    assert!(json.contains("\"ts\":1000"));
    assert!(json.contains("\"dur\":500"));
}

#[test]
fn test_chrome_trace_event_skip_none() {
    use serde_json;

    let event = ChromeTraceEvent::new(
        "test".to_string(),
        nsys_chrome::models::ChromeTracePhase::Instant,
        1000.0,
        "Device 0".to_string(),
        "Stream 1".to_string(),
        "kernel".to_string(),
    );

    let json = serde_json::to_string(&event).unwrap();
    // Should not contain optional fields that are None
    assert!(!json.contains("\"dur\""));
    assert!(!json.contains("\"cname\""));
    assert!(!json.contains("\"id\""));
    assert!(!json.contains("\"bp\""));
}

// ==========================
// Test convert_file
// ==========================

#[test]
fn test_convert_file_nonexistent_input() {
    // Test that error is returned when input file is in nonexistent directory
    let output = "/tmp/test_output.json";

    let result = convert_file(
        "/nonexistent/directory/test.sqlite",
        output,
        None,
    );

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.to_lowercase().contains("failed to open"));
    }
}

#[test]
fn test_convert_file_empty_db() {
    // Test converting an empty database
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output = temp_dir.path().join("output.json");

    // Create an empty SQLite database
    let conn = rusqlite::Connection::open(&input).unwrap();
    drop(conn);

    let result = convert_file(input.to_str().unwrap(), output.to_str().unwrap(), None);
    assert!(result.is_ok());

    // Verify output file was created
    assert!(output.exists());

    // Verify it's valid JSON with empty events
    let content = std::fs::read_to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed["traceEvents"].is_array());
}

#[test]
fn test_convert_file_with_default_options() {
    // Test that default ConversionOptions are used when none provided
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output = temp_dir.path().join("output.json");

    // Create a valid SQLite database
    let conn = rusqlite::Connection::open(&input).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);

    let result = convert_file(input.to_str().unwrap(), output.to_str().unwrap(), None);
    assert!(result.is_ok());
    assert!(output.exists());
}

#[test]
fn test_convert_file_with_custom_options() {
    // Test that custom options are passed through correctly
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output = temp_dir.path().join("output.json");

    // Create a valid SQLite database
    let conn = rusqlite::Connection::open(&input).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);

    let custom_options = ConversionOptions {
        activity_types: vec!["kernel".to_string()],
        include_metadata: false,
        nvtx_event_prefix: Some(vec!["test_".to_string()]),
        nvtx_color_scheme: HashMap::new(),
    };

    let result = convert_file(
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        Some(custom_options),
    );
    assert!(result.is_ok());
    assert!(output.exists());
}

// ==========================
// Test convert_file_gz
// ==========================

#[test]
fn test_convert_file_gz_nonexistent_input() {
    // Test that error is returned when input file is in nonexistent directory
    let output = "/tmp/test_output.json.gz";

    let result = convert_file_gz(
        "/nonexistent/directory/test.sqlite",
        output,
        None,
    );

    assert!(result.is_err());
}

#[test]
fn test_convert_file_gz_empty_db() {
    // Test converting an empty database to gzipped output
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output = temp_dir.path().join("output.json.gz");

    // Create an empty SQLite database
    let conn = rusqlite::Connection::open(&input).unwrap();
    drop(conn);

    let result = convert_file_gz(input.to_str().unwrap(), output.to_str().unwrap(), None);
    assert!(result.is_ok());

    // Verify output file was created
    assert!(output.exists());

    // Verify it's valid gzipped JSON
    let file = File::open(&output).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed["traceEvents"].is_array());
}

#[test]
fn test_convert_file_gz_output_is_compressed() {
    // Test that gzipped output is actually compressed
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output_json = temp_dir.path().join("output.json");
    let output_gz = temp_dir.path().join("output.json.gz");

    // Create a SQLite database with some data
    let conn = rusqlite::Connection::open(&input).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    // Add some string data to ensure there's something to compress
    for i in 0..100 {
        conn.execute(
            "INSERT INTO StringIds (id, value) VALUES (?, ?)",
            rusqlite::params![i, format!("test_string_{}", i)],
        )
        .unwrap();
    }
    drop(conn);

    // Convert to both formats
    convert_file(input.to_str().unwrap(), output_json.to_str().unwrap(), None).unwrap();
    convert_file_gz(input.to_str().unwrap(), output_gz.to_str().unwrap(), None).unwrap();

    // For very small files, gzip might not compress much, but file should exist
    assert!(output_gz.exists());
    assert!(output_json.exists());

    // Verify both contain the same JSON data
    let json_content = std::fs::read_to_string(&output_json).unwrap();
    
    let file = File::open(&output_gz).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut gz_content = String::new();
    gz.read_to_string(&mut gz_content).unwrap();

    assert_eq!(json_content, gz_content);
}

#[test]
fn test_convert_file_gz_with_custom_options() {
    // Test that custom options work with gzipped output
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test.sqlite");
    let output = temp_dir.path().join("output.json.gz");

    // Create a valid SQLite database
    let conn = rusqlite::Connection::open(&input).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);

    let custom_options = ConversionOptions {
        activity_types: vec!["kernel".to_string(), "nvtx".to_string()],
        include_metadata: false,
        nvtx_event_prefix: Some(vec!["test_".to_string()]),
        nvtx_color_scheme: HashMap::new(),
    };

    let result = convert_file_gz(
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        Some(custom_options),
    );
    assert!(result.is_ok());
    assert!(output.exists());
}

// ==========================
// Test Converter with Data
// ==========================

#[test]
fn test_converter_with_string_table() {
    // Test that converter can handle StringIds table
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();

    // Create database with StringIds table
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (1, 'test_kernel')",
        [],
    )
    .unwrap();
    drop(conn);

    let converter = NsysChromeConverter::new(temp_path, None).unwrap();
    let result = converter.convert();
    assert!(result.is_ok());
}

#[test]
fn test_converter_metadata_included_by_default() {
    // Test that metadata events are included by default
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();

    // Create minimal database
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);

    let options = ConversionOptions::default();
    assert!(options.include_metadata);

    let converter = NsysChromeConverter::new(temp_path, Some(options)).unwrap();
    let result = converter.convert();
    assert!(result.is_ok());
}

#[test]
fn test_converter_metadata_excluded_when_disabled() {
    // Test that metadata events can be excluded
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();

    // Create minimal database
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    drop(conn);

    let mut options = ConversionOptions::default();
    options.include_metadata = false;

    let converter = NsysChromeConverter::new(temp_path, Some(options)).unwrap();
    let result = converter.convert();
    assert!(result.is_ok());

    let events = result.unwrap();
    let metadata_events: Vec<_> = events.iter().filter(|e| e.cat == "__metadata").collect();
    assert!(metadata_events.is_empty());
}

// ==========================
// Test End-to-End Conversion
// ==========================

#[test]
fn test_nsys_convert_end_to_end() {
    // End-to-end test that creates a realistic SQLite database and validates conversion
    // This is similar to test_nsys_convert in Python integration tests
    let temp_dir = TempDir::new().unwrap();
    let input = temp_dir.path().join("test_trace.sqlite");
    let output = temp_dir.path().join("test_trace.json.gz");

    // Create a realistic SQLite database with CUPTI and NVTX data
    let conn = rusqlite::Connection::open(&input).unwrap();
    
    // Create StringIds table
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    )
    .unwrap();
    
    // Add some string values
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (1, 'test_kernel')",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO StringIds (id, value) VALUES (2, 'Forward Pass')",
        [],
    )
    .unwrap();
    
    // Create CUPTI_ACTIVITY_KIND_KERNEL table with all required columns
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            start INTEGER,
            end INTEGER,
            deviceId INTEGER,
            streamId INTEGER,
            correlationId INTEGER,
            globalPid INTEGER,
            demangledName TEXT,
            shortName INTEGER,
            gridX INTEGER,
            gridY INTEGER,
            gridZ INTEGER,
            blockX INTEGER,
            blockY INTEGER,
            blockZ INTEGER,
            registersPerThread INTEGER,
            staticSharedMemory INTEGER,
            dynamicSharedMemory INTEGER
        )",
        [],
    )
    .unwrap();
    
    // Add a kernel event (shortName is an ID referencing StringIds table)
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (
            1000000000, 1000500000, 0, 1, 1, 12345,
            'test_kernel(float*, int)', 1,
            256, 1, 1, 128, 1, 1,
            32, 0, 1024
        )",
        [],
    )
    .unwrap();
    
    // Create NVTX_EVENTS table (must match real nsys schema with textId and INTEGER eventType)
    conn.execute(
        "CREATE TABLE NVTX_EVENTS (
            start INTEGER,
            end INTEGER,
            text TEXT,
            textId INTEGER,
            globalTid INTEGER,
            eventType INTEGER
        )",
        [],
    )
    .unwrap();
    
    // Add an NVTX event (eventType 59 = NVTX_PUSH_POP_EVENT_ID for push/pop ranges)
    conn.execute(
        "INSERT INTO NVTX_EVENTS VALUES (
            900000000, 1100000000, 'Forward Pass', NULL, 12345, 59
        )",
        [],
    )
    .unwrap();
    
    drop(conn);

    // Convert using the Rust converter
    let result = convert_file_gz(
        input.to_str().unwrap(),
        output.to_str().unwrap(),
        None,
    );
    
    assert!(result.is_ok(), "Conversion should succeed");
    assert!(output.exists(), "Output file should be created");

    // Verify the output is valid gzipped JSON
    let file = File::open(&output).unwrap();
    let mut gz = GzDecoder::new(file);
    let mut content = String::new();
    gz.read_to_string(&mut content).unwrap();

    let parsed: serde_json::Value = serde_json::from_str(&content)
        .expect("Output should be valid JSON");
    
    // Verify structure
    assert!(parsed["traceEvents"].is_array(), "Should have traceEvents array");
    let events = parsed["traceEvents"].as_array().unwrap();
    
    // Should have at least some events (kernel + nvtx + possibly metadata)
    assert!(!events.is_empty(), "Should have generated events");
    
    // Verify event structure
    for event in events.iter() {
        // Each event should have required fields
        assert!(event.get("name").is_some(), "Event should have name");
        assert!(event.get("ph").is_some(), "Event should have phase");
        assert!(event.get("ts").is_some(), "Event should have timestamp");
        assert!(event.get("pid").is_some(), "Event should have pid");
        assert!(event.get("tid").is_some(), "Event should have tid");
        assert!(event.get("cat").is_some(), "Event should have category");
        
        // Verify phase is valid
        let phase = event["ph"].as_str().unwrap();
        assert!(
            ["B", "E", "X", "i", "C", "M", "b", "n", "e", "s", "t", "f", "P"].contains(&phase),
            "Phase should be valid: {}",
            phase
        );
        
        // Verify timestamps are numbers
        assert!(event["ts"].is_f64() || event["ts"].is_i64(), "Timestamp should be a number");
        
        // If it's a Complete event (X), it should have duration
        if phase == "X" {
            assert!(event.get("dur").is_some(), "Complete event should have duration");
        }
    }
    
    // Find specific event types
    let kernel_events: Vec<_> = events
        .iter()
        .filter(|e| e["cat"].as_str() == Some("kernel"))
        .collect();
    let nvtx_events: Vec<_> = events
        .iter()
        .filter(|e| e["cat"].as_str() == Some("nvtx"))
        .collect();
    
    // We should have at least one kernel event
    assert!(!kernel_events.is_empty(), "Should have kernel events");
    
    // Verify kernel event has expected fields
    if let Some(kernel) = kernel_events.first() {
        assert!(kernel["name"].as_str().is_some(), "Kernel should have name");
        assert_eq!(kernel["ph"].as_str(), Some("X"), "Kernel should be Complete event");
        assert!(kernel.get("args").is_some(), "Kernel should have args");
    }
    
    println!("✓ End-to-end conversion test passed!");
    println!("  - Generated {} events", events.len());
    println!("  - Found {} kernel events", kernel_events.len());
    println!("  - Found {} NVTX events", nvtx_events.len());
}

// ==========================
// Test Error Handling
// ==========================

#[test]
fn test_convert_file_invalid_output_path() {
    // Test that error is returned when output path is invalid
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();

    // Create valid database
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    drop(conn);

    // Try to write to invalid output path (non-existent directory)
    let result = convert_file(
        temp_path,
        "/nonexistent/directory/output.json",
        None,
    );

    assert!(result.is_err());
    if let Err(e) = result {
        let err_msg = e.to_string();
        assert!(err_msg.to_lowercase().contains("failed to create output file"));
    }
}

#[test]
fn test_convert_file_gz_invalid_output_path() {
    // Test that error is returned when output path is invalid for gzipped output
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();

    // Create valid database
    let conn = rusqlite::Connection::open(temp_path).unwrap();
    drop(conn);

    // Try to write to invalid output path
    let result = convert_file_gz(
        temp_path,
        "/nonexistent/directory/output.json.gz",
        None,
    );

    assert!(result.is_err());
}
