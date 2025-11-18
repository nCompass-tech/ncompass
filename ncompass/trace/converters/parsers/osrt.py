"""OS Runtime API event parser."""

import sqlite3
from typing import Any

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from .base import BaseParser, DefaultParserImpl


class OSRTParser(DefaultParserImpl):
    """Parser for OSRT_API table."""
    
    def __init__(self):
        DefaultParserImpl.__init__(self, "OSRT_API")
    
    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Parse OS runtime API events."""
        events = []
        
        conn.row_factory = sqlite3.Row
        query = f"SELECT start, end, globalTid, nameId, returnValue, nestingLevel FROM {self.table_name}"
        
        for row in conn.execute(query):
            if row["end"] is None:
                continue
            
            pid, tid = decompose_global_tid(row["globalTid"])
            api_name = strings.get(row["nameId"], "Unknown OS API")
            
            # Use process name or PID as process identifier
            process_name = f"Process {pid}"
            thread_name = thread_names.get(tid, f"Thread {tid}")
            
            event = ChromeTraceEvent(
                name=api_name,
                ph="X",
                cat="osrt",
                ts=ns_to_us(row["start"]),
                dur=ns_to_us(row["end"] - row["start"]),
                pid=process_name,
                tid=thread_name,
                args={
                    "returnValue": row["returnValue"],
                    "nestingLevel": row["nestingLevel"],
                }
            )
            events.append(event)
        
        return events

