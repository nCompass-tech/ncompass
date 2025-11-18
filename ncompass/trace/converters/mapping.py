"""Device and thread mapping utilities."""

import sqlite3
from typing import Optional
from collections import defaultdict

from .schema import table_exists


def decompose_global_tid(global_tid: int) -> tuple[int, int]:
    """Extract PID and TID from globalTid.
    
    nsys encodes globalTid as: globalTid = (PID << 24) | TID
    
    Args:
        global_tid: Serialized GlobalId
        
    Returns:
        Tuple of (PID, TID)
    """
    pid = (global_tid >> 24) & 0xFFFFFF
    tid = global_tid & 0xFFFFFF
    return pid, tid


def extract_device_mapping(conn: sqlite3.Connection) -> dict[int, int]:
    """Extract mapping from PID to device ID.
    
    Tries multiple methods:
    1. From CUPTI_ACTIVITY_KIND_KERNEL table (if available)
    2. From other CUPTI tables
    
    Args:
        conn: SQLite connection
        
    Returns:
        Dictionary mapping PID to device ID
    """
    pid_to_device = {}
    
    # Method 1: From CUPTI_ACTIVITY_KIND_KERNEL
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        conn.row_factory = sqlite3.Row
        for row in conn.execute(
            "SELECT DISTINCT deviceId, globalPid / 0x1000000 % 0x1000000 AS PID "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ):
            pid = row["PID"]
            device_id = row["deviceId"]
            if pid in pid_to_device:
                if pid_to_device[pid] != device_id:
                    # Warn but don't fail - some traces may have multiple devices per PID
                    pass
            else:
                pid_to_device[pid] = device_id
    
    # Method 2: From CUPTI_ACTIVITY_KIND_RUNTIME (if kernel table not available)
    elif table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        # Try to infer from runtime events - this is less reliable
        # For now, we'll return empty dict and let parsers handle it
        pass
    
    return pid_to_device


def extract_thread_names(conn: sqlite3.Connection) -> dict[int, str]:
    """Extract thread name mappings from ThreadNames table.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Dictionary mapping TID to thread name
    """
    tid_to_name = {}
    
    if table_exists(conn, "ThreadNames"):
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT * FROM ThreadNames"):
            # Column names may vary, try common ones
            if "tid" in row.keys() and "name" in row.keys():
                tid_to_name[row["tid"]] = row["name"]
            elif "globalTid" in row.keys() and "name" in row.keys():
                _, tid = decompose_global_tid(row["globalTid"])
                tid_to_name[tid] = row["name"]
    
    return tid_to_name


def get_all_devices(conn: sqlite3.Connection) -> set[int]:
    """Get all device IDs present in the trace.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Set of device IDs
    """
    devices = set()
    
    # Try CUPTI_ACTIVITY_KIND_KERNEL
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        conn.row_factory = sqlite3.Row
        for row in conn.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            devices.add(row["deviceId"])
    
    # Also check from device mapping
    device_map = extract_device_mapping(conn)
    devices.update(device_map.values())
    
    return devices

