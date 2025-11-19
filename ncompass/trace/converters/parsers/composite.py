"""Composite event parser."""

import sqlite3
from typing import Any

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from .base import BaseParser
from .default import default_init, default_table_exists, default_safe_parse


class CompositeParser(BaseParser):
    """Parser for COMPOSITE_EVENTS table."""
    
    def __init__(self):
        default_init(self, "COMPOSITE_EVENTS")
    
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
        """Parse composite events.
        
        COMPOSITE_EVENTS represent aggregated or composite events.
        The exact structure may vary, so we'll handle common fields.
        """
        events = []
        
        conn.row_factory = sqlite3.Row
        query = f"SELECT * FROM {self.table_name}"
        
        # Get column names to check what's available
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description]
        
        for row in cursor:
            # Try to extract common fields
            pid = None
            tid = None
            
            if "globalTid" in columns:
                pid, tid = decompose_global_tid(row["globalTid"])
            elif "pid" in columns and "tid" in columns:
                pid = row["pid"]
                tid = row["tid"]
            
            if pid is None or tid is None:
                continue
            
            # Create event - duration may not be available
            event_name = "Composite Event"
            process_name = f"Process {pid}"
            thread_name = thread_names.get(tid, f"Thread {tid}")
            
            # Build args from available columns
            args = {}
            for col in columns:
                if col not in ["id", "start", "end", "globalTid", "pid", "tid"]:
                    args[col] = row[col]
            
            event = ChromeTraceEvent(
                name=event_name,
                ph="X" if "end" in columns and row.get("end") else "i",
                cat="composite",
                ts=ns_to_us(row["start"]),
                pid=process_name,
                tid=thread_name,
                args=args
            )
            
            if "end" in columns and row.get("end"):
                event.dur = ns_to_us(row["end"] - row["start"])
            
            events.append(event)
        
        return events

