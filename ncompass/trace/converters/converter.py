"""Main converter class for nsys SQLite to Chrome Trace conversion."""

import sqlite3
from typing import Any, Optional
from collections import defaultdict
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
from ncompass.types import Immutable, mutate

from ncompass.trace.infra.utils import logger

from .models import ChromeTraceEvent, ConversionOptions
from .schema import detect_available_tables, TableRegistry
from .mapping import (
    extract_device_mapping,
    extract_thread_names,
    get_all_devices,
)
from .parsers import (
    CUPTIKernelParser,
    CUPTIRuntimeParser,
    NVTXParser,
    OSRTParser,
    SchedParser,
    CompositeParser,
)
from .linker import link_nvtx_to_kernels

class NsysToChromeTraceConverter(Immutable):
    """Main converter class for nsys SQLite to Chrome Trace conversion."""
    
    # def __init__(self, sqlite_path: str, options: ConversionOptions | None = None):
    def __init__(self):
        """Initialize converter.
        
        Args:
            sqlite_path: Path to input SQLite file
            options: Conversion options (defaults to all event types)
        """
        self.sqlite_path: str = ""
        self.options: ConversionOptions | None = None or ConversionOptions()
        self.conn: sqlite3.Connection | None = None
        self.strings: dict[int, str] = {}
        self.device_map: dict[int, int] = {}
        self.thread_names: dict[int, str] = {}
        self.available_tables: set[str] = set()

    @mutate
    def set_sqlite_path(self, sqlite_path: str) -> Self:
        self.sqlite_path = sqlite_path
        return self
    
    @mutate
    def set_options(self, options: ConversionOptions) -> Self:
        self.options = options 
        return self
    
    @mutate
    def __enter__(self) -> Self:
        """Context manager entry."""
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()
    
    def _load_strings(self) -> dict[int, str]:
        """Load StringIds table into dictionary.
        
        Returns:
            Dictionary mapping string ID to string value
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        strings = {}
        try:
            for row in self.conn.execute("SELECT id, value FROM StringIds"):
                strings[row["id"]] = row["value"]
        except sqlite3.OperationalError:
            # StringIds table may not exist
            pass
        
        return strings
    
    def _detect_event_types(self) -> set[str]:
        """Detect available event types based on tables.
        
        Returns:
            Set of available activity type strings
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        self.available_tables = detect_available_tables(self.conn)
        available_activities = set()
        
        for table_name in self.available_tables:
            activity_type = TableRegistry.get_activity_type(table_name)
            if activity_type:
                available_activities.add(activity_type)
        
        # nvtx-kernel is a synthetic activity type that requires kernel, cuda-api, and nvtx
        if {"kernel", "cuda-api", "nvtx"}.issubset(available_activities):
            available_activities.add("nvtx-kernel")
        
        return available_activities
    
    def _link_nvtx_to_kernels(
        self,
        nvtx_events: list[ChromeTraceEvent],
        kernel_events: list[ChromeTraceEvent],
        cuda_api_events: list[ChromeTraceEvent],
    ) -> tuple[list[ChromeTraceEvent], set[tuple], list[ChromeTraceEvent]]:
        """Link NVTX events to kernel events via CUDA API correlation.
        
        This creates nvtx-kernel events that show NVTX ranges aligned to
        actual kernel execution times, and generates flow events (arrows) 
        between CUDA API calls and their corresponding kernels.
        
        Returns:
            Tuple of:
            - nvtx-kernel events (GPU timeline)
            - mapped NVTX identifiers (for filtering)
            - flow events (arrows for visualization)
        """
        return link_nvtx_to_kernels(
            nvtx_events,
            cuda_api_events,
            kernel_events,
            self.options
        )
    
    def _parse_all_events(self) -> list[ChromeTraceEvent]:
        """Parse all events based on options and available tables.
        
        Returns:
            List of Chrome Trace events
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        events = []
        available_activities = self._detect_event_types()
        
        # Filter requested activities by what's actually available
        requested_activities = set(self.options.activity_types)
        # TODO: Should we leave this as an &?
        activities_to_parse = requested_activities & available_activities
        
        # Track parsed events for nvtx-kernel linking
        kernel_events = []
        cuda_api_events = []
        nvtx_events = []
        
        # Parse kernel events
        if "kernel" in activities_to_parse:
            parser = CUPTIKernelParser()
            kernel_events = parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names
            )
            events.extend(kernel_events)
        
        # Parse CUDA API events
        if "cuda-api" in activities_to_parse:
            parser = CUPTIRuntimeParser()
            cuda_api_events = parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names
            )
            events.extend(cuda_api_events)
        
        # Parse NVTX events
        if "nvtx" in activities_to_parse:
            parser = NVTXParser()
            nvtx_events = parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names,
            )
            events.extend(nvtx_events)
        
        # Parse nvtx-kernel events (requires linking)
        if "nvtx-kernel" in activities_to_parse:
            if kernel_events and cuda_api_events and nvtx_events:
                nvtx_kernel_events, mapped_nvtx_identifiers, flow_events = self._link_nvtx_to_kernels(
                    nvtx_events, kernel_events, cuda_api_events
                )
                events.extend(nvtx_kernel_events)
                events.extend(flow_events)
                
                # Option B: Remove mapped NVTX events from CPU timeline, keep unmapped ones
                if mapped_nvtx_identifiers:
                    # Build identifiers for nvtx_events to compare
                    unmapped_nvtx_events = []
                    for event in nvtx_events:
                        # Build identifier from event args (already stored during parsing)
                        device_id = event.args.get("deviceId")
                        tid = event.args.get("raw_tid")
                        start_ns = event.args.get("start_ns")
                        name = event.name
                        
                        event_identifier = (device_id, tid, start_ns, name)
                        if event_identifier not in mapped_nvtx_identifiers:
                            # Keep unmapped NVTX events (CPU-only work)
                            unmapped_nvtx_events.append(event)
                    
                    # Remove all NVTX events from main list, then add back only unmapped ones
                    events = [e for e in events if e.cat != "nvtx"]
                    events.extend(unmapped_nvtx_events)
            else:
                logger.warning(
                    "nvtx-kernel requested but requires kernel, cuda-api, and nvtx events. "
                    "Skipping nvtx-kernel events.",
                    UserWarning
                )
        
        # Parse OS runtime events
        if "osrt" in activities_to_parse:
            parser = OSRTParser()
            events.extend(parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names
            ))
        
        # Parse scheduling events
        if "sched" in activities_to_parse:
            parser = SchedParser()
            events.extend(parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names
            ))
        
        # Parse composite events
        if "composite" in activities_to_parse:
            parser = CompositeParser()
            events.extend(parser.safe_parse(
                self.conn, self.strings, self.options,
                self.device_map, self.thread_names
            ))
        
        return events
    
    def _add_metadata_events(self) -> list[ChromeTraceEvent]:
        """Add metadata events for process and thread names.
        
        Returns:
            List of metadata Chrome Trace events
        """
        if not self.options.include_metadata:
            return []
        
        events = []
        
        # Add process name events
        devices = get_all_devices(self.conn) if self.conn else set()
        for device_id in devices:
            event = ChromeTraceEvent(
                name="process_name",
                ph="M",
                cat="__metadata",
                ts=0.0,
                pid=f"Device {device_id}",
                tid="",
                args={"name": f"Device {device_id}"}
            )
            events.append(event)
        
        # Add thread name events (if we have thread names)
        for tid, name in self.thread_names.items():
            # We need to determine which process this thread belongs to
            # For now, we'll create events for each device
            for device_id in devices:
                event = ChromeTraceEvent(
                    name="thread_name",
                    ph="M",
                    cat="__metadata",
                    ts=0.0,
                    pid=f"Device {device_id}",
                    tid=f"Thread {tid}",
                    args={"name": name}
                )
                events.append(event)
        
        return events
    
    def _sort_events(self, events: list[ChromeTraceEvent]) -> list[ChromeTraceEvent]:
        """Sort events by timestamp, then pid, then tid.
        
        Args:
            events: List of events to sort
            
        Returns:
            Sorted list of events
        """
        return sorted(events, key=lambda e: (e.ts, e.pid, e.tid))
    
    @mutate
    def convert(self) -> list[ChromeTraceEvent]:
        """Perform the conversion.
        
        Returns:
            List of Chrome Trace events
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        # Load required data
        self.strings = self._load_strings()
        self.device_map = extract_device_mapping(self.conn)
        self.thread_names = extract_thread_names(self.conn)
        
        
        # Parse all events
        events = self._parse_all_events()
        
        # Add metadata events
        if self.options.include_metadata:
            events.extend(self._add_metadata_events())
        
        # Sort events
        events = self._sort_events(events)
        
        return events

def convert_file(
    sqlite_path: str,
    output_path: str,
    options: ConversionOptions | None = None
) -> None:
    """Convert nsys SQLite file to Chrome Trace JSON.
    
    Args:
        sqlite_path: Path to input SQLite file
        output_path: Path to output JSON file
        options: Conversion options
    """
    from .utils import write_chrome_trace

    converter_ctx = NsysToChromeTraceConverter()\
                        .set_sqlite_path(sqlite_path)\
                        .set_options(options)
    
    with converter_ctx as converter:
        events = converter.convert()
        # Convert Pydantic models to dicts
        event_dicts = [event.to_dict() for event in events]
        write_chrome_trace(output_path, event_dicts)

