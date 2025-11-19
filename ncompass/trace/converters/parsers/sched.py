"""Thread scheduling event parser."""

import sqlite3
from typing import Any

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from .base import BaseParser
from .default import default_init, default_table_exists, default_safe_parse


class SchedParser(BaseParser):
    """Parser for SCHED_EVENTS table."""
    
    def __init__(self):
        default_init(self, "SCHED_EVENTS")
    
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
        """Parse thread scheduling events.
        
        SCHED_EVENTS represent thread state changes (scheduled in/out).
        We'll create events showing when threads are scheduled.
        """
        events = []
        
        conn.row_factory = sqlite3.Row
        query = f"SELECT start, cpu, isSchedIn, globalTid, threadState, threadBlock FROM {self.table_name}"
        
        for row in conn.execute(query):
            pid, tid = decompose_global_tid(row["globalTid"])
            
            # Create instant event for scheduling change
            event_name = "Scheduled In" if row["isSchedIn"] else "Scheduled Out"
            process_name = f"Process {pid}"
            thread_name = thread_names.get(tid, f"Thread {tid}")
            
            event = ChromeTraceEvent(
                name=event_name,
                ph="i",  # Instant event
                cat="sched",
                ts=ns_to_us(row["start"]),
                pid=process_name,
                tid=thread_name,
                args={
                    "cpu": row["cpu"],
                    "threadState": row["threadState"],
                    "threadBlock": row["threadBlock"],
                }
            )
            events.append(event)
        
        return events

