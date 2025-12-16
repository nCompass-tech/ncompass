"""Main converter class for nsys SQLite to Chrome Trace conversion."""

import sqlite3
import sys
from typing import Any, Optional, Iterator
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
        
        available_tables = detect_available_tables(self.conn)
        available_activities = set()
        
        for table_name in available_tables:
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
    
    def _parse_all_events(
        self, 
        strings: dict[int, str],
        device_map: dict[int, int],
        thread_names: dict[int, str]
    ) -> list[ChromeTraceEvent]:
        """Parse all events based on options and available tables.
        
        Args:
            strings: Dictionary mapping string ID to string value
            device_map: Dictionary mapping device IDs
            thread_names: Dictionary mapping thread IDs to names
            
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
                self.conn, strings, self.options,
                device_map, thread_names
            )
            events.extend(kernel_events)
        
        # Parse CUDA API events
        if "cuda-api" in activities_to_parse:
            parser = CUPTIRuntimeParser()
            cuda_api_events = parser.safe_parse(
                self.conn, strings, self.options,
                device_map, thread_names
            )
            events.extend(cuda_api_events)
        
        # Parse NVTX events
        if "nvtx" in activities_to_parse:
            parser = NVTXParser()
            nvtx_events = parser.safe_parse(
                self.conn, strings, self.options,
                device_map, thread_names,
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
                    "Skipping nvtx-kernel events."
                )
        
        # Parse OS runtime events
        if "osrt" in activities_to_parse:
            parser = OSRTParser()
            events.extend(parser.safe_parse(
                self.conn, strings, self.options,
                device_map, thread_names
            ))
        
        # Parse scheduling events
        if "sched" in activities_to_parse:
            parser = SchedParser()
            events.extend(parser.safe_parse(
                self.conn, strings, self.options,
                device_map, thread_names
            ))
        
        # Parse composite events
        if "composite" in activities_to_parse:
            parser = CompositeParser()
            events.extend(parser.safe_parse(
                self.conn, strings, self.options,
                device_map, thread_names
            ))
        
        return events
    
    def _add_metadata_events(self, thread_names: dict[int, str]) -> list[ChromeTraceEvent]:
        """Add metadata events for process and thread names.
        
        Args:
            thread_names: Dictionary mapping thread IDs to names
            
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
        for tid, name in thread_names.items():
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
    
    def convert(self) -> Iterator[dict]:
        """Perform the conversion, yielding events as a stream.
        
        Yields:
            Chrome Trace event dictionaries
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")
        
        # Load required data
        strings = self._load_strings()
        device_map = extract_device_mapping(self.conn)
        thread_names = extract_thread_names(self.conn)
        
        # Parse all events
        events = self._parse_all_events(strings, device_map, thread_names)
        
        # Add metadata events
        if self.options.include_metadata:
            events.extend(self._add_metadata_events(thread_names))
        
        # Sort events
        events = self._sort_events(events)
        
        # Yield events one at a time
        for event in events:
            yield event.to_dict()

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
        event_stream = converter.convert()
        # Stream events directly to file
        write_chrome_trace(output_path, event_stream)


def convert_nsys_report(
    nsys_rep_path: str,
    output_path: str,
    options: ConversionOptions | None = None,
    keep_sqlite: bool = False,
    use_rust: bool = True,
) -> None:
    """Convert nsys report (.nsys-rep) to gzip-compressed Chrome Trace JSON.
    
    This function performs the full conversion pipeline:
    1. Converts .nsys-rep to SQLite using nsys CLI (if using Python)
    2. Converts SQLite to Chrome Trace format
    3. Writes output as gzip-compressed JSON (.json.gz)
    
    Args:
        nsys_rep_path: Path to input nsys report file (.nsys-rep)
        output_path: Path to output gzip-compressed JSON file (.json.gz)
        options: Conversion options (defaults to common activity types)
        keep_sqlite: If True, keep the intermediate SQLite file
        use_rust: If True, use the Rust implementation (default: True, faster)
        
    Raises:
        FileNotFoundError: If input file doesn't exist or nsys CLI not found
        subprocess.CalledProcessError: If nsys export fails
        RuntimeError: If conversion fails
    """
    import subprocess
    import tempfile
    from pathlib import Path
    from .utils import write_chrome_trace_gz

    nsys_rep_file = Path(nsys_rep_path)
    
    # Validate input file exists
    if not nsys_rep_file.exists():
        raise FileNotFoundError(f"Input file not found: {nsys_rep_path}")
    
    # Use Rust implementation if available and requested
    if use_rust:
        import ncompass
        ncompass_pkg_dir = Path(ncompass.__file__).parent
        
        # Try packaged binary location first (installed via pip)
        rust_binary = ncompass_pkg_dir / "bin" / "nsys-chrome"
        is_dev_binary = False
        
        # Fallback to development location (editable install / dev checkout)
        if not rust_binary.exists():
            ncompass_root = ncompass_pkg_dir.parent
            rust_binary = (
                ncompass_root /
                "ncompass_rust" /
                "trace_converters" /
                "target" /
                "x86_64-unknown-linux-musl" /
                "release" /
                "nsys-chrome"
            )
            is_dev_binary = True

        if rust_binary.exists():
            if is_dev_binary:
                logger.info(
                    "Using Rust binary from development build. "
                    "For production, install with: pip install ncompass"
                )
            # Build command line arguments
            cmd = [str(rust_binary), str(nsys_rep_path), "-o", str(output_path)]
            
            # Add activity types if specified
            if options is not None and options.activity_types:
                cmd.extend(["-t", ",".join(options.activity_types)])
            
            # Add NVTX prefix filter if specified
            if options is not None and options.nvtx_event_prefix:
                cmd.extend(["--nvtx-prefix", ",".join(options.nvtx_event_prefix)])
            
            # Add metadata flag
            if options is not None and not options.include_metadata:
                cmd.append("--metadata=false")
            
            # Add keep-sqlite flag
            if keep_sqlite:
                cmd.append("--keep-sqlite")
            
            try:
                print(f"Running comand => {cmd}")
                result = subprocess.run(cmd, check=True, text=True)
                # Print stderr to show progress messages
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                return
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"Rust converter failed: {e.stderr}\nFalling back to Python implementation."
                )
                # Fall through to Python implementation
            except Exception as e:
                logger.warning(
                    f"Error running Rust converter: {e}\nFalling back to Python implementation."
                )
                # Fall through to Python implementation
        else:
            logger.warning(
                f"Rust binary not found at {rust_binary}. "
                "Using Python implementation. "
                "Build the Rust version for speedup: cd ncompass_rust/trace_converters && "
                "cargo build --release --target=x86_64-unknown-linux-musl"
            )
    
    # Python implementation (original code)
    # Determine SQLite file path
    if keep_sqlite:
        sqlite_path = nsys_rep_file.with_suffix('.sqlite')
    else:
        # Use temp file that will be cleaned up
        temp_dir = tempfile.gettempdir()
        sqlite_path = Path(temp_dir) / f"{nsys_rep_file.stem}.sqlite"
    
    try:
        # Step 1: Convert nsys-rep to SQLite using nsys CLI
        export_command = [
            "nsys", "export",
            "--type", "sqlite",
            "--force-overwrite", "true",
            "-o", str(sqlite_path),
            str(nsys_rep_file)
        ]
        
        try:
            subprocess.run(export_command, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                "'nsys' command not found. Please ensure nsys CLI is installed "
                "and available in your PATH."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nsys export failed: {e.stderr}") from e
        
        # Step 2 & 3: Convert SQLite to Chrome Trace and write as gzip
        if options is None:
            options = ConversionOptions(
                activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
                include_metadata=True
            )
        
        converter_ctx = NsysToChromeTraceConverter()\
                            .set_sqlite_path(str(sqlite_path))\
                            .set_options(options)
        
        with converter_ctx as converter:
            event_stream = converter.convert()
            write_chrome_trace_gz(output_path, event_stream)
    
    finally:
        # Clean up SQLite file if not keeping it
        if not keep_sqlite and sqlite_path.exists():
            sqlite_path.unlink()
