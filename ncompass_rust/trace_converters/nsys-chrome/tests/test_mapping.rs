//! Unit tests for mapping module

use nsys_chrome::mapping::{decompose_global_tid, extract_device_mapping, extract_thread_names, get_all_devices};
use rusqlite::Connection;
use tempfile::NamedTempFile;

// ==========================
// Tests for decompose_global_tid
// ==========================

#[test]
fn test_decompose_global_tid_basic() {
    // globalTid = (PID << 24) | TID
    // PID = 1, TID = 100 -> globalTid = (1 << 24) | 100 = 16777316
    let global_tid: i64 = (1 << 24) | 100;
    let (pid, tid) = decompose_global_tid(global_tid);
    assert_eq!(pid, 1);
    assert_eq!(tid, 100);
}

#[test]
fn test_decompose_global_tid_zero() {
    let (pid, tid) = decompose_global_tid(0);
    assert_eq!(pid, 0);
    assert_eq!(tid, 0);
}

#[test]
fn test_decompose_global_tid_max_tid() {
    // Max TID is 0xFFFFFF (24 bits)
    let global_tid: i64 = (1 << 24) | 0xFFFFFF;
    let (pid, tid) = decompose_global_tid(global_tid);
    assert_eq!(pid, 1);
    assert_eq!(tid, 0xFFFFFF);
}

#[test]
fn test_decompose_global_tid_max_pid() {
    // Max PID is also 0xFFFFFF (24 bits extracted)
    let global_tid: i64 = (0xFFFFFF_i64 << 24) | 100;
    let (pid, tid) = decompose_global_tid(global_tid);
    assert_eq!(pid, 0xFFFFFF);
    assert_eq!(tid, 100);
}

#[test]
fn test_decompose_global_tid_large_value() {
    // Test with a realistic large value
    let global_tid: i64 = 12345678901234;
    let (pid, tid) = decompose_global_tid(global_tid);
    
    // Verify the decomposition is reversible
    let reconstructed = ((pid as i64) << 24) | (tid as i64);
    // Due to masking, we only get lower 48 bits
    let expected = global_tid & 0xFFFFFFFFFFFF;
    assert_eq!(reconstructed, expected);
}

#[test]
fn test_decompose_global_tid_only_tid() {
    // Only TID set, PID is 0
    let global_tid: i64 = 12345;
    let (pid, tid) = decompose_global_tid(global_tid);
    assert_eq!(pid, 0);
    assert_eq!(tid, 12345);
}

#[test]
fn test_decompose_global_tid_typical_values() {
    // Typical nsys values
    let test_cases = vec![
        ((100 << 24) | 1, 100, 1),
        ((100 << 24) | 2, 100, 2),
        ((101 << 24) | 1, 101, 1),
    ];

    for (global_tid, expected_pid, expected_tid) in test_cases {
        let (pid, tid) = decompose_global_tid(global_tid);
        assert_eq!(pid, expected_pid, "Failed for global_tid {}", global_tid);
        assert_eq!(tid, expected_tid, "Failed for global_tid {}", global_tid);
    }
}

// ==========================
// Tests for extract_device_mapping
// ==========================

#[test]
fn test_extract_device_mapping_empty_db() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Empty database - no tables
    let result = extract_device_mapping(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_extract_device_mapping_no_kernel_table() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create some other table but not kernel table
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    ).unwrap();
    
    let result = extract_device_mapping(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_extract_device_mapping_with_kernel_table() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create kernel table with required columns
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER,
            globalPid INTEGER
        )",
        [],
    ).unwrap();
    
    // Insert test data
    // PID = 100, deviceId = 0
    let global_pid_0: i64 = 100 * 0x1000000; // PID 100
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![0, global_pid_0],
    ).unwrap();
    
    // PID = 101, deviceId = 1
    let global_pid_1: i64 = 101 * 0x1000000; // PID 101
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![1, global_pid_1],
    ).unwrap();
    
    let result = extract_device_mapping(&conn).unwrap();
    
    assert_eq!(result.len(), 2);
    assert_eq!(result.get(&100), Some(&0));
    assert_eq!(result.get(&101), Some(&1));
}

#[test]
fn test_extract_device_mapping_duplicate_entries() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create kernel table
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER,
            globalPid INTEGER
        )",
        [],
    ).unwrap();
    
    // Insert duplicate PIDs with same device
    let global_pid: i64 = 100 * 0x1000000;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![0, global_pid],
    ).unwrap();
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![0, global_pid],
    ).unwrap();
    
    let result = extract_device_mapping(&conn).unwrap();
    
    // Should deduplicate
    assert_eq!(result.len(), 1);
    assert_eq!(result.get(&100), Some(&0));
}

// ==========================
// Tests for extract_thread_names
// ==========================

#[test]
fn test_extract_thread_names_empty_db() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    let result = extract_thread_names(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_extract_thread_names_no_table() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create some other table
    conn.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)",
        [],
    ).unwrap();
    
    let result = extract_thread_names(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_extract_thread_names_with_tid_column() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create ThreadNames table with tid column
    conn.execute(
        "CREATE TABLE ThreadNames (tid INTEGER, name TEXT)",
        [],
    ).unwrap();
    
    conn.execute(
        "INSERT INTO ThreadNames (tid, name) VALUES (?, ?)",
        rusqlite::params![1, "Main Thread"],
    ).unwrap();
    conn.execute(
        "INSERT INTO ThreadNames (tid, name) VALUES (?, ?)",
        rusqlite::params![2, "Worker Thread"],
    ).unwrap();
    
    let result = extract_thread_names(&conn).unwrap();
    
    assert_eq!(result.len(), 2);
    assert_eq!(result.get(&1), Some(&"Main Thread".to_string()));
    assert_eq!(result.get(&2), Some(&"Worker Thread".to_string()));
}

#[test]
fn test_extract_thread_names_with_global_tid_column() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create ThreadNames table with globalTid column
    conn.execute(
        "CREATE TABLE ThreadNames (globalTid INTEGER, name TEXT)",
        [],
    ).unwrap();
    
    // globalTid = (PID << 24) | TID
    let global_tid_1: i64 = (100 << 24) | 1;
    let global_tid_2: i64 = (100 << 24) | 2;
    
    conn.execute(
        "INSERT INTO ThreadNames (globalTid, name) VALUES (?, ?)",
        rusqlite::params![global_tid_1, "Main Thread"],
    ).unwrap();
    conn.execute(
        "INSERT INTO ThreadNames (globalTid, name) VALUES (?, ?)",
        rusqlite::params![global_tid_2, "Worker Thread"],
    ).unwrap();
    
    let result = extract_thread_names(&conn).unwrap();
    
    assert_eq!(result.len(), 2);
    assert_eq!(result.get(&1), Some(&"Main Thread".to_string()));
    assert_eq!(result.get(&2), Some(&"Worker Thread".to_string()));
}

#[test]
fn test_extract_thread_names_missing_name_column() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create ThreadNames table without name column
    conn.execute(
        "CREATE TABLE ThreadNames (tid INTEGER, other_column TEXT)",
        [],
    ).unwrap();
    
    let result = extract_thread_names(&conn).unwrap();
    
    // Should return empty map if no 'name' column
    assert!(result.is_empty());
}

// ==========================
// Tests for get_all_devices
// ==========================

#[test]
fn test_get_all_devices_empty_db() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    let result = get_all_devices(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_get_all_devices_from_kernel_table() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create kernel table
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER,
            globalPid INTEGER
        )",
        [],
    ).unwrap();
    
    // Insert data for multiple devices
    let global_pid: i64 = 100 * 0x1000000;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![0, global_pid],
    ).unwrap();
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![1, global_pid],
    ).unwrap();
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![2, global_pid],
    ).unwrap();
    
    let result = get_all_devices(&conn).unwrap();
    
    assert_eq!(result.len(), 3);
    assert!(result.contains(&0));
    assert!(result.contains(&1));
    assert!(result.contains(&2));
}

#[test]
fn test_get_all_devices_deduplication() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create kernel table
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER,
            globalPid INTEGER
        )",
        [],
    ).unwrap();
    
    // Insert multiple entries for same device
    let global_pid: i64 = 100 * 0x1000000;
    for _ in 0..5 {
        conn.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
            rusqlite::params![0, global_pid],
        ).unwrap();
    }
    
    let result = get_all_devices(&conn).unwrap();
    
    // Should deduplicate to single device
    assert_eq!(result.len(), 1);
    assert!(result.contains(&0));
}

#[test]
fn test_get_all_devices_sorted() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();
    
    // Create kernel table
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            deviceId INTEGER,
            globalPid INTEGER
        )",
        [],
    ).unwrap();
    
    // Insert in non-sorted order
    let global_pid: i64 = 100 * 0x1000000;
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![3, global_pid],
    ).unwrap();
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![1, global_pid],
    ).unwrap();
    conn.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (deviceId, globalPid) VALUES (?, ?)",
        rusqlite::params![2, global_pid],
    ).unwrap();
    
    let result = get_all_devices(&conn).unwrap();
    
    // Should be sorted
    assert_eq!(result, vec![1, 2, 3]);
}

