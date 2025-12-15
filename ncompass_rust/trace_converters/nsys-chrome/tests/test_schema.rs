//! Unit tests for schema module

use nsys_chrome::schema::{detect_available_tables, detect_event_types, table_exists, TableRegistry};
use rusqlite::Connection;
use tempfile::NamedTempFile;

// ==========================
// Tests for TableRegistry
// ==========================

#[test]
fn test_table_registry_get_activity_type_kernel() {
    let result = TableRegistry::get_activity_type("CUPTI_ACTIVITY_KIND_KERNEL");
    assert_eq!(result, Some("kernel"));
}

#[test]
fn test_table_registry_get_activity_type_runtime() {
    let result = TableRegistry::get_activity_type("CUPTI_ACTIVITY_KIND_RUNTIME");
    assert_eq!(result, Some("cuda-api"));
}

#[test]
fn test_table_registry_get_activity_type_nvtx() {
    let result = TableRegistry::get_activity_type("NVTX_EVENTS");
    assert_eq!(result, Some("nvtx"));
}

#[test]
fn test_table_registry_get_activity_type_osrt() {
    let result = TableRegistry::get_activity_type("OSRT_API");
    assert_eq!(result, Some("osrt"));
}

#[test]
fn test_table_registry_get_activity_type_sched() {
    let result = TableRegistry::get_activity_type("SCHED_EVENTS");
    assert_eq!(result, Some("sched"));
}

#[test]
fn test_table_registry_get_activity_type_composite() {
    let result = TableRegistry::get_activity_type("COMPOSITE_EVENTS");
    assert_eq!(result, Some("composite"));
}

#[test]
fn test_table_registry_get_activity_type_unknown() {
    let result = TableRegistry::get_activity_type("UNKNOWN_TABLE");
    assert_eq!(result, None);
}

#[test]
fn test_table_registry_get_activity_type_empty() {
    let result = TableRegistry::get_activity_type("");
    assert_eq!(result, None);
}

#[test]
fn test_table_registry_get_tables_for_activity_kernel() {
    let result = TableRegistry::get_tables_for_activity("kernel");
    assert_eq!(result, vec!["CUPTI_ACTIVITY_KIND_KERNEL"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_cuda_api() {
    let result = TableRegistry::get_tables_for_activity("cuda-api");
    assert_eq!(result, vec!["CUPTI_ACTIVITY_KIND_RUNTIME"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_nvtx() {
    let result = TableRegistry::get_tables_for_activity("nvtx");
    assert_eq!(result, vec!["NVTX_EVENTS"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_osrt() {
    let result = TableRegistry::get_tables_for_activity("osrt");
    assert_eq!(result, vec!["OSRT_API"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_sched() {
    let result = TableRegistry::get_tables_for_activity("sched");
    assert_eq!(result, vec!["SCHED_EVENTS"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_composite() {
    let result = TableRegistry::get_tables_for_activity("composite");
    assert_eq!(result, vec!["COMPOSITE_EVENTS"]);
}

#[test]
fn test_table_registry_get_tables_for_activity_unknown() {
    let result = TableRegistry::get_tables_for_activity("unknown");
    assert!(result.is_empty());
}

// ==========================
// Tests for table_exists
// ==========================

#[test]
fn test_table_exists_true() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE test_table (id INTEGER PRIMARY KEY)",
        [],
    )
    .unwrap();

    let result = table_exists(&conn, "test_table").unwrap();
    assert!(result);
}

#[test]
fn test_table_exists_false() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    let result = table_exists(&conn, "nonexistent_table").unwrap();
    assert!(!result);
}

#[test]
fn test_table_exists_case_sensitive() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE TestTable (id INTEGER PRIMARY KEY)",
        [],
    )
    .unwrap();

    // sqlite_master stores exact table name, so lookups are case-sensitive
    let result1 = table_exists(&conn, "TestTable").unwrap();
    let result2 = table_exists(&conn, "testtable").unwrap();
    let result3 = table_exists(&conn, "TESTTABLE").unwrap();

    assert!(result1);  // Exact match - found
    assert!(!result2); // Wrong case - not found
    assert!(!result3); // Wrong case - not found
}

#[test]
fn test_table_exists_empty_database() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    let result = table_exists(&conn, "any_table").unwrap();
    assert!(!result);
}

#[test]
fn test_table_exists_multiple_tables() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute("CREATE TABLE table1 (id INTEGER PRIMARY KEY)", [])
        .unwrap();
    conn.execute("CREATE TABLE table2 (id INTEGER PRIMARY KEY)", [])
        .unwrap();
    conn.execute("CREATE TABLE table3 (id INTEGER PRIMARY KEY)", [])
        .unwrap();

    assert!(table_exists(&conn, "table1").unwrap());
    assert!(table_exists(&conn, "table2").unwrap());
    assert!(table_exists(&conn, "table3").unwrap());
    assert!(!table_exists(&conn, "table4").unwrap());
}

// ==========================
// Tests for detect_available_tables
// ==========================

#[test]
fn test_detect_available_tables_empty_db() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    let result = detect_available_tables(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_detect_available_tables_single_table() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE test_table (id INTEGER PRIMARY KEY)",
        [],
    )
    .unwrap();

    let result = detect_available_tables(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("test_table"));
}

#[test]
fn test_detect_available_tables_multiple_tables() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute("CREATE TABLE table1 (id INTEGER PRIMARY KEY)", [])
        .unwrap();
    conn.execute("CREATE TABLE table2 (id INTEGER PRIMARY KEY)", [])
        .unwrap();
    conn.execute("CREATE TABLE table3 (id INTEGER PRIMARY KEY)", [])
        .unwrap();

    let result = detect_available_tables(&conn).unwrap();
    assert_eq!(result.len(), 3);
    assert!(result.contains("table1"));
    assert!(result.contains("table2"));
    assert!(result.contains("table3"));
}

#[test]
fn test_detect_available_tables_nsys_tables() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    // Create typical nsys tables
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();
    conn.execute("CREATE TABLE StringIds (id INTEGER)", [])
        .unwrap();

    let result = detect_available_tables(&conn).unwrap();
    assert_eq!(result.len(), 4);
    assert!(result.contains("CUPTI_ACTIVITY_KIND_KERNEL"));
    assert!(result.contains("CUPTI_ACTIVITY_KIND_RUNTIME"));
    assert!(result.contains("NVTX_EVENTS"));
    assert!(result.contains("StringIds"));
}

// ==========================
// Tests for detect_event_types
// ==========================

#[test]
fn test_detect_event_types_empty_db() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_detect_event_types_kernel_only() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("kernel"));
}

#[test]
fn test_detect_event_types_cuda_api_only() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("cuda-api"));
}

#[test]
fn test_detect_event_types_nvtx_only() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("nvtx"));
}

#[test]
fn test_detect_event_types_osrt_only() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute("CREATE TABLE OSRT_API (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("osrt"));
}

#[test]
fn test_detect_event_types_sched_only() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute("CREATE TABLE SCHED_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();
    assert_eq!(result.len(), 1);
    assert!(result.contains("sched"));
}

#[test]
fn test_detect_event_types_nvtx_kernel_synthetic() {
    // nvtx-kernel is a synthetic type requiring kernel, cuda-api, and nvtx
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();

    // Should have kernel, cuda-api, nvtx, AND nvtx-kernel
    assert_eq!(result.len(), 4);
    assert!(result.contains("kernel"));
    assert!(result.contains("cuda-api"));
    assert!(result.contains("nvtx"));
    assert!(result.contains("nvtx-kernel"));
}

#[test]
fn test_detect_event_types_nvtx_kernel_missing_kernel() {
    // nvtx-kernel should NOT be present if kernel is missing
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains("cuda-api"));
    assert!(result.contains("nvtx"));
    assert!(!result.contains("nvtx-kernel"));
}

#[test]
fn test_detect_event_types_nvtx_kernel_missing_cuda_api() {
    // nvtx-kernel should NOT be present if cuda-api is missing
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains("kernel"));
    assert!(result.contains("nvtx"));
    assert!(!result.contains("nvtx-kernel"));
}

#[test]
fn test_detect_event_types_nvtx_kernel_missing_nvtx() {
    // nvtx-kernel should NOT be present if nvtx is missing
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();

    let result = detect_event_types(&conn).unwrap();

    assert_eq!(result.len(), 2);
    assert!(result.contains("kernel"));
    assert!(result.contains("cuda-api"));
    assert!(!result.contains("nvtx-kernel"));
}

#[test]
fn test_detect_event_types_all_tables() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    // Create all known tables
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE NVTX_EVENTS (id INTEGER)", [])
        .unwrap();
    conn.execute("CREATE TABLE OSRT_API (id INTEGER)", [])
        .unwrap();
    conn.execute("CREATE TABLE SCHED_EVENTS (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();

    // Should have all 6 types (including synthetic nvtx-kernel)
    assert_eq!(result.len(), 6);
    assert!(result.contains("kernel"));
    assert!(result.contains("cuda-api"));
    assert!(result.contains("nvtx"));
    assert!(result.contains("osrt"));
    assert!(result.contains("sched"));
    assert!(result.contains("nvtx-kernel"));
}

#[test]
fn test_detect_event_types_unrecognized_tables_ignored() {
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap();
    let conn = Connection::open(temp_path).unwrap();

    // Create mix of recognized and unrecognized tables
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (id INTEGER)",
        [],
    )
    .unwrap();
    conn.execute("CREATE TABLE StringIds (id INTEGER)", [])
        .unwrap();
    conn.execute("CREATE TABLE UnknownTable (id INTEGER)", [])
        .unwrap();

    let result = detect_event_types(&conn).unwrap();

    // Should only include recognized activity types
    assert_eq!(result.len(), 1);
    assert!(result.contains("kernel"));
}

