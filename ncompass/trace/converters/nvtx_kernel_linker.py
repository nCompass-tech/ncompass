"""Link NVTX events to kernel events via CUDA API correlation.

This module implements the logic to link NVTX markers to kernel execution times
by finding overlapping CUDA API calls and using correlationId to connect them.
"""

from collections import defaultdict
from typing import Any, Optional
import re

from .models import ChromeTraceEvent, ConversionOptions
from .utils import ns_to_us


def _extract_event_data(event: ChromeTraceEvent) -> dict[str, Any]:
    """Extract raw data from ChromeTraceEvent for linking logic.
    
    Args:
        event: ChromeTraceEvent object
        
    Returns:
        Dictionary with fields compatible with DB row format:
        - start: Start timestamp in nanoseconds
        - end: End timestamp in nanoseconds
        - deviceId: Device ID
        - correlationId: Correlation ID (for kernels/CUDA API)
        - name: Event name
        - tid: Raw thread ID
        - pid: Raw process ID
        - _original_event: Reference to original ChromeTraceEvent
    """
    return {
        "start": event.args.get("start_ns"),
        "end": event.args.get("end_ns"),
        "deviceId": event.args.get("deviceId"),
        "correlationId": event.args.get("correlationId"),
        "name": event.name,
        "tid": event.args.get("raw_tid"),
        "pid": event.args.get("raw_pid"),
        "_original_event": event,
    }

def _create_flow_events(
    cuda_api_event: ChromeTraceEvent,
    kernel_event: ChromeTraceEvent,
    correlation_id: int,
) -> tuple[ChromeTraceEvent, ChromeTraceEvent]:
    """Create flow start/end events to show arrows in Perfetto.
    
    Flow events link CUDA API calls to their corresponding kernel executions,
    rendering as arrows in the trace viewer.
    
    Args:
        cuda_api_event: The CUDA API call event (flow source)
        kernel_event: The kernel execution event (flow destination)
        correlation_id: Correlation ID linking the two events
        
    Returns:
        Tuple of (flow_start, flow_finish) ChromeTraceEvent objects
    """
    # Flow start: at CUDA API event
    flow_start = ChromeTraceEvent(
        name="",  # Empty name for flow events
        ph="s",   # Flow start phase
        cat="cuda_flow",
        ts=cuda_api_event.ts,
        pid=cuda_api_event.pid,
        tid=cuda_api_event.tid,
        id=correlation_id,  # Links the flow
        args={}
    )
    
    # Flow finish: at kernel event
    flow_finish = ChromeTraceEvent(
        name="",
        ph="f",   # Flow finish phase
        cat="cuda_flow",
        ts=kernel_event.ts,
        pid=kernel_event.pid,
        tid=kernel_event.tid,
        id=correlation_id,  # Same ID links it to the start
        bp="e",  # Binding point: enclosing slice
        args={}
    )
    
    return flow_start, flow_finish

def _find_overlapping_intervals(nvtx_rows: list, cuda_api_rows: list) -> list[tuple]:
    """Find which CUDA API calls overlap with each NVTX range.
    
    Uses a sweep-line algorithm to efficiently find overlapping intervals.
    This matches the implementation from the reference script.
    
    Args:
        nvtx_rows: List of NVTX event rows (with 'start' and 'end' fields)
        cuda_api_rows: List of CUDA API event rows (with 'start' and 'end' fields)
        
    Returns:
        List of (nvtx_row, overlapping_cuda_api_rows) tuples
    """
    mixed_rows = []
    
    # Create index mapping for nvtx_rows (indices are hashable)
    nvtx_index_map = {id(nvtx_row): i for i, nvtx_row in enumerate(nvtx_rows)}
    
    # Add NVTX events as start/end pairs
    for nvtx_row in nvtx_rows:
        start = nvtx_row["start"]
        end = nvtx_row["end"]
        mixed_rows.append((start, 1, "nvtx", nvtx_row))
        mixed_rows.append((end, -1, "nvtx", nvtx_row))
    
    # Add CUDA API events as start/end pairs
    for cuda_api_row in cuda_api_rows:
        start = cuda_api_row["start"]
        end = cuda_api_row["end"]
        mixed_rows.append((start, 1, "cuda_api", cuda_api_row))
        mixed_rows.append((end, -1, "cuda_api", cuda_api_row))
    
    # Sort by timestamp, then by event type (start=1 before end=-1), then by origin
    mixed_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    
    active_nvtx_intervals = []
    result_by_index = defaultdict(list)  # Use index as key instead of dict
    
    for timestamp, event_type, event_origin, orig_event in mixed_rows:
        if event_type == 1:  # Start event
            if event_origin == "nvtx":
                active_nvtx_intervals.append(orig_event)
            else:  # cuda_api start
                # Add this CUDA API call to all currently active NVTX ranges
                for nvtx_event in active_nvtx_intervals:
                    nvtx_idx = nvtx_index_map[id(nvtx_event)]
                    result_by_index[nvtx_idx].append(orig_event)
        else:  # End event (event_type == -1)
            if event_origin == "nvtx":
                active_nvtx_intervals.remove(orig_event)
    
    # Convert index-based result to list of tuples
    return [(nvtx_rows[idx], cuda_api_list) 
            for idx, cuda_api_list in result_by_index.items()]

def link_nvtx_to_kernels(
    nvtx_events: list[ChromeTraceEvent],
    cuda_api_events: list[ChromeTraceEvent],
    kernel_events: list[ChromeTraceEvent],
    options: ConversionOptions,
) -> tuple[list[ChromeTraceEvent], set[tuple], list[ChromeTraceEvent]]:
    """Link NVTX events to kernel events via CUDA API correlation.
    
    This function works on already-parsed ChromeTraceEvent objects and:
    1. Groups events by device ID
    2. For each device, finds overlapping intervals between NVTX and CUDA API events
    3. Uses correlationId to link CUDA API calls to kernels
    4. Creates new "nvtx-kernel" events that snap NVTX markers to kernel timelines
    5. Generates flow events (arrows) between all CUDA API calls and their kernels
    
    Args:
        nvtx_events: Parsed NVTX events
        cuda_api_events: Parsed CUDA API events
        kernel_events: Parsed kernel events
        options: Conversion options (for color scheme)
        
    Returns:
        Tuple of:
        - nvtx-kernel events (GPU timeline showing NVTX-annotated work)
        - mapped_nvtx_identifiers (to filter out original NVTX from CPU timeline)
        - flow events (arrows between CUDA API and kernels)
    """
    nvtx_kernel_events = []
    mapped_nvtx_identifiers = set()
    flow_events = []
    
    # Group events by device ID
    per_device_nvtx = defaultdict(list)
    per_device_cuda_api = defaultdict(list)
    per_device_kernels = defaultdict(list)
    
    for event in nvtx_events:
        device_id = event.args.get("deviceId")
        start_ns = event.args.get("start_ns")
        end_ns = event.args.get("end_ns")
        # Skip incomplete NVTX ranges (missing start or end timestamp)
        if device_id is not None and start_ns is not None and end_ns is not None:
            per_device_nvtx[device_id].append(_extract_event_data(event))
    
    for event in cuda_api_events:
        device_id = event.args.get("deviceId")
        corr_id = event.args.get("correlationId")
        # Skip events without device ID or correlation ID (needed for linking)
        if device_id is not None and corr_id is not None:
            per_device_cuda_api[device_id].append(_extract_event_data(event))
    
    for event in kernel_events:
        device_id = event.args.get("deviceId")
        corr_id = event.args.get("correlationId")
        # Skip events without device ID or correlation ID (needed for linking)
        if device_id is not None and corr_id is not None:
            per_device_kernels[device_id].append(_extract_event_data(event))
    
    # Get devices that have all three event types
    common_devices = (
        set(per_device_nvtx.keys()) & 
        set(per_device_cuda_api.keys()) & 
        set(per_device_kernels.keys())
    )
    
    # Process each device
    for device_id in common_devices:
        nvtx_rows = per_device_nvtx[device_id]
        cuda_api_rows = per_device_cuda_api[device_id]
        kernel_rows = per_device_kernels[device_id]
        
        # Find overlapping intervals between NVTX and CUDA API events
        event_map = _find_overlapping_intervals(nvtx_rows, cuda_api_rows)
        
        # Build correlationId -> {cuda_api, kernels[]} mapping
        # NOTE: Multiple kernels can share the same correlationId (e.g., CUDA graphs)
        correlation_id_map = defaultdict(lambda: {"cuda_api": None, "kernels": []})
        
        # Map CUDA API rows by correlationId
        for cuda_api_row in cuda_api_rows:
            corr_id = cuda_api_row["correlationId"]
            if corr_id is not None:
                correlation_id_map[corr_id]["cuda_api"] = cuda_api_row
        
        # Map kernel rows by correlationId (APPEND to support multiple kernels)
        for kernel_row in kernel_rows:
            corr_id = kernel_row["correlationId"]
            if corr_id is not None:
                correlation_id_map[corr_id]["kernels"].append(kernel_row)
        
        # Generate flow events for ALL CUDA API → Kernel links (Option A)
        for corr_id, data in correlation_id_map.items():
            cuda_api = data["cuda_api"]
            kernels = data["kernels"]
            
            if cuda_api is not None and len(kernels) > 0:
                cuda_api_orig = cuda_api["_original_event"]
                
                # Create flow arrow to EACH kernel (handles cudaGraphLaunch → multiple kernels)
                for kernel_row in kernels:
                    kernel_orig = kernel_row["_original_event"]
                    
                    flow_start, flow_finish = _create_flow_events(
                        cuda_api_orig,
                        kernel_orig,
                        corr_id
                    )
                    flow_events.extend([flow_start, flow_finish])
        
        # For each NVTX event with overlapping CUDA API calls, find kernels
        for nvtx_row, cuda_api_rows_overlapping in event_map:
            kernel_start_time = None
            kernel_end_time = None
            
            nvtx_name = nvtx_row["name"]
            
            # Find kernels via correlationId from overlapping CUDA API calls
            for cuda_api_row in cuda_api_rows_overlapping:
                corr_id = cuda_api_row["correlationId"]
                if corr_id is None or corr_id not in correlation_id_map:
                    continue
                
                kernels = correlation_id_map[corr_id]["kernels"]
                if len(kernels) == 0:
                    # CUDA API call didn't launch a kernel, skip
                    continue
                
                # Process ALL kernels with this correlationId
                for kernel_row in kernels:
                    # Track min start and max end times across ALL kernels
                    if kernel_start_time is None or kernel_start_time > kernel_row["start"]:
                        kernel_start_time = kernel_row["start"]
                    if kernel_end_time is None or kernel_end_time < kernel_row["end"]:
                        kernel_end_time = kernel_row["end"]
            
            # Create nvtx-kernel event if we found kernels
            if kernel_start_time is not None and kernel_end_time is not None:
                tid = nvtx_row["tid"]
                
                event = ChromeTraceEvent(
                    name=nvtx_name or "",
                    ph="X",
                    cat="nvtx-kernel",
                    ts=ns_to_us(kernel_start_time),
                    dur=ns_to_us(kernel_end_time - kernel_start_time),
                    pid=f"Device {device_id}",
                    tid=f"NVTX Kernel Thread {tid}",
                    args={}
                )
                
                # Apply color scheme if specified
                if options.nvtx_color_scheme:
                    for key, color in options.nvtx_color_scheme.items():
                        if re.search(key, nvtx_name):
                            event.cname = color
                            break
                
                nvtx_kernel_events.append(event)
                
                # Track this NVTX event as successfully mapped (for Option B filtering)
                nvtx_identifier = (device_id, tid, nvtx_row["start"], nvtx_name)
                mapped_nvtx_identifiers.add(nvtx_identifier)
    
    return nvtx_kernel_events, mapped_nvtx_identifiers, flow_events

