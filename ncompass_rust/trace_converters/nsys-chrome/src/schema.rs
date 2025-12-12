//! Schema detection and table discovery for nsys SQLite databases

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashSet;

/// Detect all available tables in the SQLite database
pub fn detect_available_tables(conn: &Connection) -> Result<HashSet<String>> {
    let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'")?;
    let tables = stmt
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<Result<HashSet<String>, _>>()?;
    Ok(tables)
}

/// Check if a table exists in the database
pub fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?1"
    )?;
    let mut rows = stmt.query([table_name])?;
    Ok(rows.next()?.is_some())
}

/// Registry mapping table names to activity types
pub struct TableRegistry;

impl TableRegistry {
    /// Get activity type for a table name
    pub fn get_activity_type(table_name: &str) -> Option<&'static str> {
        match table_name {
            "CUPTI_ACTIVITY_KIND_KERNEL" => Some("kernel"),
            "CUPTI_ACTIVITY_KIND_RUNTIME" => Some("cuda-api"),
            "NVTX_EVENTS" => Some("nvtx"),
            "OSRT_API" => Some("osrt"),
            "SCHED_EVENTS" => Some("sched"),
            "COMPOSITE_EVENTS" => Some("composite"),
            _ => None,
        }
    }

    /// Get table names for an activity type
    pub fn get_tables_for_activity(activity_type: &str) -> Vec<&'static str> {
        match activity_type {
            "kernel" => vec!["CUPTI_ACTIVITY_KIND_KERNEL"],
            "cuda-api" => vec!["CUPTI_ACTIVITY_KIND_RUNTIME"],
            "nvtx" => vec!["NVTX_EVENTS"],
            "osrt" => vec!["OSRT_API"],
            "sched" => vec!["SCHED_EVENTS"],
            "composite" => vec!["COMPOSITE_EVENTS"],
            _ => vec![],
        }
    }
}

/// Detect available event types based on tables
pub fn detect_event_types(conn: &Connection) -> Result<HashSet<String>> {
    let available_tables = detect_available_tables(conn)?;
    let mut available_activities = HashSet::new();

    for table_name in &available_tables {
        if let Some(activity_type) = TableRegistry::get_activity_type(table_name) {
            available_activities.insert(activity_type.to_string());
        }
    }

    // nvtx-kernel is a synthetic activity type that requires kernel, cuda-api, and nvtx
    if available_activities.contains("kernel")
        && available_activities.contains("cuda-api")
        && available_activities.contains("nvtx")
    {
        available_activities.insert("nvtx-kernel".to_string());
    }

    Ok(available_activities)
}

