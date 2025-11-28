"""CUPTI event parsers for CUDA kernel and runtime events."""

import sqlite3
from typing import Any

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from .base import BaseParser
from .default import default_init, default_table_exists, default_safe_parse


class CUPTIKernelParser(BaseParser):
    """Parser for CUPTI_ACTIVITY_KIND_KERNEL table."""
    
    def __init__(self):
        default_init(self, "CUPTI_ACTIVITY_KIND_KERNEL")
    
    def table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check if the table exists in the database."""
        return default_table_exists(self, conn)
    
    def safe_parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Safely parse events, returning empty list if table doesn't exist."""
        return default_safe_parse(self, conn, strings, options, device_map, thread_names)
    
    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Parse CUDA kernel events."""
        events = []
        
        conn.row_factory = sqlite3.Row
        for row in conn.execute(f"SELECT * FROM {self.table_name}"):
            device_id = row["deviceId"]
            stream_id = row["streamId"]
            kernel_name = strings.get(row["shortName"], "Unknown Kernel")
            
            event = ChromeTraceEvent(
                name=kernel_name,
                ph="X",
                cat="kernel",
                ts=ns_to_us(row["start"]),
                dur=ns_to_us(row["end"] - row["start"]),
                pid=f"Device {device_id}",
                tid=f"Stream {stream_id}",
                args={
                    "grid": [row["gridX"], row["gridY"], row["gridZ"]],
                    "block": [row["blockX"], row["blockY"], row["blockZ"]],
                    "registersPerThread": row["registersPerThread"],
                    "staticSharedMemory": row["staticSharedMemory"],
                    "dynamicSharedMemory": row["dynamicSharedMemory"],
                    "correlationId": row["correlationId"],
                    "deviceId": device_id,
                    "streamId": stream_id,
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                }
            )
            events.append(event)
        
        return events


class CUPTIRuntimeParser(BaseParser):
    """Parser for CUPTI_ACTIVITY_KIND_RUNTIME table."""
    
    def __init__(self):
        default_init(self, "CUPTI_ACTIVITY_KIND_RUNTIME")
    
    def table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check if the table exists in the database."""
        return default_table_exists(self, conn)
    
    def safe_parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Safely parse events, returning empty list if table doesn't exist."""
        return default_safe_parse(self, conn, strings, options, device_map, thread_names)
    
    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Parse CUDA runtime API events."""
        events = []
        
        conn.row_factory = sqlite3.Row
        query = (
            f"SELECT start, end, globalTid, correlationId, nameId "
            f"FROM {self.table_name}"
        )
        
        for row in conn.execute(query):
            pid, tid = decompose_global_tid(row["globalTid"])
            device_id = device_map.get(pid)
            
            if device_id is None:
                # If we can't determine device, use PID as fallback
                device_id = pid
            
            api_name = strings.get(row["nameId"], "Unknown API")
            
            event = ChromeTraceEvent(
                name=api_name,
                ph="X",
                cat="cuda_api",
                ts=ns_to_us(row["start"]),
                dur=ns_to_us(row["end"] - row["start"]),
                pid=f"Device {device_id}",
                tid=f"CUDA API Thread {tid}",
                args={
                    "correlationId": row["correlationId"],
                    "deviceId": device_id,
                    "raw_pid": pid,
                    "raw_tid": tid,
                    "start_ns": row["start"],
                    "end_ns": row["end"],
                }
            )
            events.append(event)
        
        return events

