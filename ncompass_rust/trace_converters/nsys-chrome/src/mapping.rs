//! Device and thread mapping utilities

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

use crate::schema::table_exists;

/// Extract PID and TID from globalTid
///
/// nsys encodes globalTid as: globalTid = (PID << 24) | TID
pub fn decompose_global_tid(global_tid: i64) -> (i32, i32) {
    let pid = ((global_tid >> 24) & 0xFFFFFF) as i32;
    let tid = (global_tid & 0xFFFFFF) as i32;
    (pid, tid)
}

/// Extract mapping from PID to device ID
///
/// Tries multiple methods:
/// 1. From CUPTI_ACTIVITY_KIND_KERNEL table (if available)
/// 2. From other CUPTI tables
pub fn extract_device_mapping(conn: &Connection) -> Result<HashMap<i32, i32>> {
    let mut pid_to_device = HashMap::default();

    // Method 1: From CUPTI_ACTIVITY_KIND_KERNEL
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL")? {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT deviceId, globalPid / 0x1000000 % 0x1000000 AS PID \
             FROM CUPTI_ACTIVITY_KIND_KERNEL",
        )?;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let device_id: i32 = row.get(0)?;
            let pid: i32 = row.get(1)?;

            if let Some(&existing_device) = pid_to_device.get(&pid) {
                if existing_device != device_id {
                    // Warn but don't fail - some traces may have multiple devices per PID
                    eprintln!(
                        "Warning: PID {} mapped to multiple devices ({} and {})",
                        pid, existing_device, device_id
                    );
                }
            } else {
                pid_to_device.insert(pid, device_id);
            }
        }
    }
    // Method 2: From CUPTI_ACTIVITY_KIND_RUNTIME (if kernel table not available)
    else if table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME")? {
        // Try to infer from runtime events - this is less reliable
        // For now, we'll return empty map and let parsers handle it
    }

    Ok(pid_to_device)
}

/// Extract thread name mappings from ThreadNames table
pub fn extract_thread_names(conn: &Connection) -> Result<HashMap<i32, String>> {
    let mut tid_to_name = HashMap::default();

    if !table_exists(conn, "ThreadNames")? {
        return Ok(tid_to_name);
    }

    // Try to detect column names
    let stmt = conn.prepare("SELECT * FROM ThreadNames LIMIT 1")?;
    let column_names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Determine which columns to use
    let has_tid = column_names.contains(&"tid".to_string());
    let has_global_tid = column_names.contains(&"globalTid".to_string());
    let has_name = column_names.contains(&"name".to_string());

    if !has_name {
        return Ok(tid_to_name);
    }

    if has_tid {
        let mut stmt = conn.prepare("SELECT tid, name FROM ThreadNames")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let tid: i32 = row.get(0)?;
            let name: String = row.get(1)?;
            tid_to_name.insert(tid, name);
        }
    } else if has_global_tid {
        let mut stmt = conn.prepare("SELECT globalTid, name FROM ThreadNames")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let global_tid: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let (_, tid) = decompose_global_tid(global_tid);
            tid_to_name.insert(tid, name);
        }
    }

    Ok(tid_to_name)
}

/// Get all device IDs present in the trace
pub fn get_all_devices(conn: &Connection) -> Result<Vec<i32>> {
    let mut devices = Vec::new();

    // Try CUPTI_ACTIVITY_KIND_KERNEL
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL")? {
        let mut stmt = conn.prepare("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let device_id: i32 = row.get(0)?;
            if !devices.contains(&device_id) {
                devices.push(device_id);
            }
        }
    }

    // Also check from device mapping
    let device_map = extract_device_mapping(conn)?;
    for device_id in device_map.values() {
        if !devices.contains(device_id) {
            devices.push(*device_id);
        }
    }

    devices.sort_unstable();
    Ok(devices)
}

