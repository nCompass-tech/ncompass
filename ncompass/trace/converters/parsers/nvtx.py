"""NVTX event parser."""

import sqlite3
import re
from typing import Any, Optional

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from ..mapping import decompose_global_tid
from .base import BaseParser
from .default import default_init, default_table_exists, default_safe_parse

class NVTXParser(BaseParser):
    """Parser for NVTX_EVENTS table."""
    
    def __init__(self):
        default_init(self, "NVTX_EVENTS")
        self.NVTX_PUSH_POP_EVENT_ID = 59
    
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
    
    def _build_filter_clause(self, event_prefix: Optional[list[str]]) -> str:
        """Build SQL WHERE clause for event prefix filtering."""
        if not event_prefix:
            return ""
        
        if len(event_prefix) == 1:
            return f" AND text LIKE '{event_prefix[0]}%'"
        
        conditions = " OR ".join(f"text LIKE '{prefix}%'" for prefix in event_prefix)
        return f" AND ({conditions})"

    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        f"""Parse NVTX events.
        
        Focuses on eventType {self.NVTX_PUSH_POP_EVENT_ID} (NvtxPushPopRange) which corresponds to
        torch.cuda.nvtx.range APIs.
        """
        events = []
        
        filter_clause = self._build_filter_clause(options.nvtx_event_prefix)
        
        conn.row_factory = sqlite3.Row
        query = (
            f"SELECT start, end, text, textId, globalTid, eventType "
            f"FROM {self.table_name} "
            f"WHERE eventType == {self.NVTX_PUSH_POP_EVENT_ID}{filter_clause}"
        )

        for row in conn.execute(query):
            if row["end"] is None:
                # Skip incomplete events
                continue
            
            pid, tid = decompose_global_tid(row["globalTid"])
            device_id = device_map.get(pid)
            
            if device_id is None:
                device_id = pid  # Fallback

            # Resolve text: prefer textId lookup, fallback to text column, then "[No name]"
            text_id = row["textId"]
            if text_id is not None:
                text = strings.get(text_id, f"[Unknown textId: {text_id}]")
            elif row["text"] is not None:
                text = row["text"]
            else:
                text = "[No name]" 
            
            # Use timestamps directly (nsys already synchronizes them)
            ts_ns = row["start"]
            end_ns = row["end"]
            ts_us = ns_to_us(ts_ns)
            dur_us = ns_to_us(end_ns - ts_ns)
            
            event = ChromeTraceEvent(
                name=text,
                ph="X",
                cat="nvtx",
                ts=ts_us,
                dur=dur_us,
                pid=f"Device {device_id}",
                tid=f"NVTX Thread {tid}",
                args={
                    "deviceId": device_id,
                    "raw_pid": pid,
                    "raw_tid": tid,
                    "start_ns": ts_ns,
                    "end_ns": end_ns,
                }
            )
            
            # Apply color scheme if specified
            if options.nvtx_color_scheme:
                for key, color in options.nvtx_color_scheme.items():
                    if re.search(key, text):
                        event.cname = color
                        break
            
            events.append(event)

        return events

