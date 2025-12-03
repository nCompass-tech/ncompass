"""SQL-based NVTX to kernel linking for memory-efficient streaming conversion.

This module provides SQL-based alternatives to the Python-based nvtx_linker.py
for linking NVTX events to kernel events. By performing the linking in SQLite,
we avoid materializing all events in Python memory.

The SQL logic is equivalent to the Python logic in nvtx_linker.py:

1. OVERLAP DETECTION:
   - Python (sweep-line algorithm): A CUDA API event is considered overlapping with
     an NVTX event if the CUDA API START occurs while the NVTX is active.
     Condition: nvtx.start <= cuda_api.start < nvtx.end
   - SQL: c.start >= n.start AND c.start < n.end
   
2. DEVICE MATCHING:
   - Python: Events are grouped by deviceId (derived from PID via device_map)
   - SQL: Match by PID extracted from globalTid (same PID = same device)
     Condition: ((c.globalTid >> 24) & 0xFFFFFF) = ((n.globalTid >> 24) & 0xFFFFFF)

3. CORRELATION MATCHING:
   - Python: CUDA API correlationId matched to kernel correlationId
   - SQL: k.correlationId = c.correlationId

4. TIME AGGREGATION:
   - Python: aggregate_kernel_times() finds MIN(start), MAX(end) across all kernels
   - SQL: MIN(k.start), MAX(k.end) with GROUP BY nvtx.rowid

5. NVTX EVENT TYPE FILTERING:
   - Python: eventType == 59 (NVTX_PUSH_POP_EVENT_ID for NvtxPushPopRange)
   - SQL: WHERE n.eventType = 59
"""

import sqlite3
import re
from typing import Iterator, Optional

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from ..schema import table_exists


# NVTX event type for push/pop ranges (torch.cuda.nvtx.range)
NVTX_PUSH_POP_EVENT_ID = 59


def _get_nvtx_kernel_query() -> str:
    """Return the SQL query for nvtx-kernel linking.
    
    This query performs the equivalent of the Python nvtx_linker logic:
    1. Find CUDA API events whose START falls within NVTX time range
    2. Match CUDA API to kernels via correlationId  
    3. Aggregate kernel times (MIN start, MAX end) per NVTX event
    
    Returns:
        SQL query string
    """
    return """
    SELECT 
        n.rowid as nvtx_rowid,
        n.start as nvtx_start,
        n.end as nvtx_end,
        n.text as nvtx_text,
        n.textId as nvtx_textId,
        n.globalTid as nvtx_globalTid,
        MIN(k.start) as kernel_start,
        MAX(k.end) as kernel_end,
        k.deviceId as device_id,
        COUNT(*) as kernel_count
    FROM NVTX_EVENTS n
    -- Join CUDA API events where API START is within NVTX range
    -- This matches the sweep-line algorithm: target added when target starts and source is active
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME c ON (
        c.start >= n.start AND c.start < n.end
        -- Same process (PID) ensures same device
        -- globalTid encoding: (PID << 24) | TID
        AND ((c.globalTid >> 24) & 0xFFFFFF) = ((n.globalTid >> 24) & 0xFFFFFF)
    )
    -- Join kernels via correlation ID
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON (
        k.correlationId = c.correlationId
    )
    -- Only process NVTX push/pop range events (eventType 59)
    WHERE n.eventType = ?
    -- Group by NVTX event to aggregate kernel times
    GROUP BY n.rowid
    """


def _get_flow_events_query() -> str:
    """Return the SQL query for CUDA API to kernel flow events.
    
    Flow events create visual arrows in Perfetto/Chrome trace viewer
    linking CUDA API calls to their corresponding kernel executions.
    
    Returns:
        SQL query string
    """
    return """
    SELECT 
        c.start as cuda_api_start,
        c.end as cuda_api_end,
        c.globalTid as cuda_api_globalTid,
        c.correlationId as correlation_id,
        k.start as kernel_start,
        k.deviceId as kernel_device_id,
        k.streamId as kernel_stream_id
    FROM CUPTI_ACTIVITY_KIND_RUNTIME c
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON (
        k.correlationId = c.correlationId
    )
    """


def _get_mapped_nvtx_query() -> str:
    """Return the SQL query to get NVTX events that have kernel mappings.
    
    These identifiers are used to filter out NVTX events from the CPU timeline
    that have been successfully mapped to nvtx-kernel events on the GPU timeline.
    
    Returns:
        SQL query string
    """
    return """
    SELECT DISTINCT
        n.start as nvtx_start,
        n.text as nvtx_text,
        n.textId as nvtx_textId,
        n.globalTid as nvtx_globalTid,
        k.deviceId as device_id
    FROM NVTX_EVENTS n
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME c ON (
        c.start >= n.start AND c.start < n.end
        AND ((c.globalTid >> 24) & 0xFFFFFF) = ((n.globalTid >> 24) & 0xFFFFFF)
    )
    JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON (
        k.correlationId = c.correlationId
    )
    WHERE n.eventType = ?
    """


def can_use_sql_linking(conn: sqlite3.Connection) -> bool:
    """Check if SQL-based linking can be used.
    
    Requires all three tables: NVTX_EVENTS, CUPTI_ACTIVITY_KIND_RUNTIME,
    and CUPTI_ACTIVITY_KIND_KERNEL.
    
    Args:
        conn: SQLite connection
        
    Returns:
        True if all required tables exist
    """
    return (
        table_exists(conn, "NVTX_EVENTS") and
        table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") and
        table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL")
    )


def stream_nvtx_kernel_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    options: ConversionOptions,
    device_map: dict[int, int],
) -> Iterator[ChromeTraceEvent]:
    """Stream nvtx-kernel events using SQL-based linking.
    
    This is a memory-efficient alternative to link_nvtx_to_kernels() that
    performs the linking in SQLite and streams results one at a time.
    
    Args:
        conn: SQLite connection
        strings: String ID to string value mapping
        options: Conversion options (for color scheme)
        device_map: PID to device ID mapping
        
    Yields:
        ChromeTraceEvent objects for nvtx-kernel events
    """
    if not can_use_sql_linking(conn):
        return
    
    conn.row_factory = sqlite3.Row
    query = _get_nvtx_kernel_query()
    
    for row in conn.execute(query, (NVTX_PUSH_POP_EVENT_ID,)):
        # Resolve text: prefer textId lookup, fallback to text column
        text_id = row["nvtx_textId"]
        nvtx_text = row["nvtx_text"]
        
        if text_id is not None:
            text = strings.get(text_id, f"[Unknown textId: {text_id}]")
        elif nvtx_text is not None:
            text = nvtx_text
        else:
            text = "[No name]"
        
        # Get thread ID from globalTid
        _, tid = decompose_global_tid(row["nvtx_globalTid"])
        device_id = row["device_id"]
        
        # Create nvtx-kernel event with kernel time range
        event = ChromeTraceEvent(
            name=text,
            ph="X",
            cat="nvtx-kernel",
            ts=ns_to_us(row["kernel_start"]),
            dur=ns_to_us(row["kernel_end"] - row["kernel_start"]),
            pid=f"Device {device_id}",
            tid=f"NVTX Kernel Thread {tid}",
            args={}
        )
        
        # Apply color scheme if specified
        if options.nvtx_color_scheme:
            for pattern, color in options.nvtx_color_scheme.items():
                if re.search(pattern, text):
                    event.cname = color
                    break
        
        yield event


def stream_flow_events(
    conn: sqlite3.Connection,
    device_map: dict[int, int],
) -> Iterator[ChromeTraceEvent]:
    """Stream flow events linking CUDA API calls to kernel executions.
    
    Flow events render as arrows in Perfetto/Chrome trace viewer,
    showing the connection between API calls and their kernel launches.
    
    Args:
        conn: SQLite connection
        device_map: PID to device ID mapping
        
    Yields:
        ChromeTraceEvent objects for flow events (start and finish pairs)
    """
    if not (table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME") and
            table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL")):
        return
    
    conn.row_factory = sqlite3.Row
    query = _get_flow_events_query()
    
    for row in conn.execute(query):
        correlation_id = row["correlation_id"]
        
        # Get CUDA API event info
        cuda_api_ts = ns_to_us(row["cuda_api_start"])
        pid, tid = decompose_global_tid(row["cuda_api_globalTid"])
        cuda_api_device_id = device_map.get(pid, pid)
        
        # Get kernel event info
        kernel_ts = ns_to_us(row["kernel_start"])
        kernel_device_id = row["kernel_device_id"]
        kernel_stream_id = row["kernel_stream_id"]
        
        # Flow start: at CUDA API event
        yield ChromeTraceEvent(
            name="",
            ph="s",  # Flow start
            cat="cuda_flow",
            ts=cuda_api_ts,
            pid=f"Device {cuda_api_device_id}",
            tid=f"CUDA API Thread {tid}",
            id=correlation_id,
            args={}
        )
        
        # Flow finish: at kernel event
        yield ChromeTraceEvent(
            name="",
            ph="f",  # Flow finish
            cat="cuda_flow",
            ts=kernel_ts,
            pid=f"Device {kernel_device_id}",
            tid=f"Stream {kernel_stream_id}",
            id=correlation_id,
            bp="e",  # Binding point: enclosing slice
            args={}
        )


def get_mapped_nvtx_identifiers(
    conn: sqlite3.Connection,
    strings: dict[int, str],
) -> set[tuple]:
    """Get identifiers of NVTX events that have been mapped to kernels.
    
    These identifiers can be used to filter out NVTX events from the
    CPU timeline that have corresponding nvtx-kernel events.
    
    Args:
        conn: SQLite connection
        strings: String ID to string value mapping
        
    Returns:
        Set of (device_id, tid, start_ns, name) tuples
    """
    if not can_use_sql_linking(conn):
        return set()
    
    mapped_identifiers = set()
    conn.row_factory = sqlite3.Row
    query = _get_mapped_nvtx_query()
    
    for row in conn.execute(query, (NVTX_PUSH_POP_EVENT_ID,)):
        # Resolve text
        text_id = row["nvtx_textId"]
        nvtx_text = row["nvtx_text"]
        
        if text_id is not None:
            text = strings.get(text_id, f"[Unknown textId: {text_id}]")
        elif nvtx_text is not None:
            text = nvtx_text
        else:
            text = "[No name]"
        
        _, tid = decompose_global_tid(row["nvtx_globalTid"])
        device_id = row["device_id"]
        start_ns = row["nvtx_start"]
        
        # This matches the identifier format from nvtx_linker.py:
        # nvtx_identifier = (device_id, tid, start_ns, nvtx_name)
        mapped_identifiers.add((device_id, tid, start_ns, text))
    
    return mapped_identifiers


def stream_unmapped_nvtx_events(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    options: ConversionOptions,
    device_map: dict[int, int],
    mapped_identifiers: set[tuple],
) -> Iterator[ChromeTraceEvent]:
    """Stream NVTX events that were NOT mapped to kernels.
    
    These are NVTX events that should remain on the CPU timeline because
    they don't have corresponding kernel executions.
    
    Args:
        conn: SQLite connection
        strings: String ID to string value mapping
        options: Conversion options
        device_map: PID to device ID mapping
        mapped_identifiers: Set of (device_id, tid, start_ns, name) tuples to exclude
        
    Yields:
        ChromeTraceEvent objects for unmapped NVTX events
    """
    if not table_exists(conn, "NVTX_EVENTS"):
        return
    
    conn.row_factory = sqlite3.Row
    query = """
    SELECT start, end, text, textId, globalTid
    FROM NVTX_EVENTS
    WHERE eventType = ?
    """
    
    # Build filter from options if specified
    filter_clause = ""
    if options.nvtx_event_prefix:
        if len(options.nvtx_event_prefix) == 1:
            filter_clause = f" AND text LIKE '{options.nvtx_event_prefix[0]}%'"
        else:
            conditions = " OR ".join(f"text LIKE '{p}%'" for p in options.nvtx_event_prefix)
            filter_clause = f" AND ({conditions})"
    
    query += filter_clause
    
    for row in conn.execute(query, (NVTX_PUSH_POP_EVENT_ID,)):
        if row["end"] is None:
            continue
        
        # Resolve text
        text_id = row["textId"]
        nvtx_text = row["text"]
        
        if text_id is not None:
            text = strings.get(text_id, f"[Unknown textId: {text_id}]")
        elif nvtx_text is not None:
            text = nvtx_text
        else:
            text = "[No name]"
        
        pid, tid = decompose_global_tid(row["globalTid"])
        device_id = device_map.get(pid, pid)
        start_ns = row["start"]
        end_ns = row["end"]
        
        # Check if this NVTX event was mapped
        identifier = (device_id, tid, start_ns, text)
        if identifier in mapped_identifiers:
            continue  # Skip - this event has a corresponding nvtx-kernel event
        
        event = ChromeTraceEvent(
            name=text,
            ph="X",
            cat="nvtx",
            ts=ns_to_us(start_ns),
            dur=ns_to_us(end_ns - start_ns),
            pid=f"Device {device_id}",
            tid=f"NVTX Thread {tid}",
            args={
                "deviceId": device_id,
                "raw_pid": pid,
                "raw_tid": tid,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )
        
        # Apply color scheme if specified
        if options.nvtx_color_scheme:
            for pattern, color in options.nvtx_color_scheme.items():
                if re.search(pattern, text):
                    event.cname = color
                    break
        
        yield event

