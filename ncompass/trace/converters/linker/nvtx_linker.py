"""Link NVTX events to kernel events via CUDA API correlation.

This module uses shared algorithms from linker.algorithms to link NVTX markers
to kernel execution times by finding overlapping CUDA API calls and using
correlationId to connect them.
"""

from collections import defaultdict
from typing import Any
import re

from ..models import ChromeTraceEvent, ConversionOptions
from ..utils import ns_to_us
from .algorithms import (
    find_overlapping_intervals,
    build_correlation_map,
    aggregate_kernel_times,
    find_kernels_for_annotation,
)
from .adapters import NsysTraceEventAdapter


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


def _group_events_by_device(
    nvtx_events: list[ChromeTraceEvent],
    cuda_api_events: list[ChromeTraceEvent],
    kernel_events: list[ChromeTraceEvent],
) -> tuple[dict[int, list[ChromeTraceEvent]], dict[int, list[ChromeTraceEvent]], dict[int, list[ChromeTraceEvent]]]:
    """Group events by device ID.
    
    Args:
        nvtx_events: Parsed NVTX events
        cuda_api_events: Parsed CUDA API events
        kernel_events: Parsed kernel events
        
    Returns:
        Tuple of (per_device_nvtx, per_device_cuda_api, per_device_kernels) dictionaries
    """
    per_device_nvtx = defaultdict(list)
    per_device_cuda_api = defaultdict(list)
    per_device_kernels = defaultdict(list)
    
    for event in nvtx_events:
        device_id = event.args.get("deviceId")
        start_ns = event.args.get("start_ns")
        end_ns = event.args.get("end_ns")
        # Skip incomplete NVTX ranges (missing start or end timestamp)
        if device_id is not None and start_ns is not None and end_ns is not None:
            per_device_nvtx[device_id].append(event)
    
    for event in cuda_api_events:
        device_id = event.args.get("deviceId")
        corr_id = event.args.get("correlationId")
        # Skip events without device ID or correlation ID (needed for linking)
        if device_id is not None and corr_id is not None:
            per_device_cuda_api[device_id].append(event)
    
    for event in kernel_events:
        device_id = event.args.get("deviceId")
        corr_id = event.args.get("correlationId")
        # Skip events without device ID or correlation ID (needed for linking)
        if device_id is not None and corr_id is not None:
            per_device_kernels[device_id].append(event)
    
    return per_device_nvtx, per_device_cuda_api, per_device_kernels


def _build_correlation_map_with_cuda_api(
    cuda_api_events_list: list[ChromeTraceEvent],
    kernel_events_list: list[ChromeTraceEvent],
    adapter: NsysTraceEventAdapter,
) -> dict[int, dict[str, Any]]:
    """Build correlation ID map including both CUDA API and kernel events.
    
    Args:
        cuda_api_events_list: List of CUDA API events for a device
        kernel_events_list: List of kernel events for a device
        adapter: Adapter to extract correlation IDs
        
    Returns:
        Dictionary mapping correlation ID to {"cuda_api": event, "kernels": [events]}
    """
    correlation_id_map = defaultdict(lambda: {"cuda_api": None, "kernels": []})
    
    # Map CUDA API events by correlationId
    for cuda_api_event in cuda_api_events_list:
        corr_id = adapter.get_correlation_id(cuda_api_event)
        if corr_id is not None:
            correlation_id_map[corr_id]["cuda_api"] = cuda_api_event
    
    # Map kernel events by correlationId (APPEND to support multiple kernels)
    kernel_correlation_map = build_correlation_map(kernel_events_list, adapter)
    for corr_id, kernels in kernel_correlation_map.items():
        correlation_id_map[corr_id]["kernels"] = kernels
    
    return correlation_id_map


def _generate_flow_events_for_correlation_map(
    correlation_id_map: dict[int, dict[str, Any]],
) -> list[ChromeTraceEvent]:
    """Generate flow events for all CUDA API → Kernel links.
    
    Args:
        correlation_id_map: Mapping from correlation ID to CUDA API and kernels
        
    Returns:
        List of flow events (start and finish pairs)
    """
    flow_events = []
    
    for corr_id, data in correlation_id_map.items():
        cuda_api_event = data["cuda_api"]
        kernels = data["kernels"]
        
        if cuda_api_event is not None and len(kernels) > 0:
            # Create flow arrow to EACH kernel (handles cudaGraphLaunch → multiple kernels)
            for kernel_event in kernels:
                flow_start, flow_finish = _create_flow_events(
                    cuda_api_event,
                    kernel_event,
                    corr_id
                )
                flow_events.extend([flow_start, flow_finish])
    
    return flow_events


def _create_nvtx_kernel_event(
    nvtx_event: ChromeTraceEvent,
    kernel_start_time: float,
    kernel_end_time: float,
    device_id: int,
    options: ConversionOptions,
) -> ChromeTraceEvent:
    """Create a single nvtx-kernel event from an NVTX event and kernel time range.
    
    Args:
        nvtx_event: The NVTX event to convert
        kernel_start_time: Start time of aggregated kernels (nanoseconds)
        kernel_end_time: End time of aggregated kernels (nanoseconds)
        device_id: Device ID
        options: Conversion options (for color scheme)
        
    Returns:
        ChromeTraceEvent for the nvtx-kernel timeline
    """
    nvtx_name = nvtx_event.name
    tid = nvtx_event.args.get("raw_tid")
    
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
    
    return event


def _process_device_nvtx_events(
    nvtx_events_list: list[ChromeTraceEvent],
    cuda_api_events_list: list[ChromeTraceEvent],
    kernel_events_list: list[ChromeTraceEvent],
    device_id: int,
    adapter: NsysTraceEventAdapter,
    options: ConversionOptions,
) -> tuple[list[ChromeTraceEvent], set[tuple], list[ChromeTraceEvent]]:
    """Process NVTX events for a single device.
    
    Args:
        nvtx_events_list: NVTX events for this device
        cuda_api_events_list: CUDA API events for this device
        kernel_events_list: Kernel events for this device
        device_id: Device ID being processed
        adapter: Adapter for extracting event data
        options: Conversion options
        
    Returns:
        Tuple of (nvtx_kernel_events, mapped_nvtx_identifiers, flow_events)
    """
    nvtx_kernel_events = []
    mapped_nvtx_identifiers = set()
    
    # Find overlapping intervals between NVTX and CUDA API events
    overlap_map = find_overlapping_intervals(
        nvtx_events_list, cuda_api_events_list, adapter, "nvtx", "cuda_api"
    )
    
    # Build correlation ID map
    correlation_id_map = _build_correlation_map_with_cuda_api(
        cuda_api_events_list, kernel_events_list, adapter
    )
    
    # Generate flow events
    flow_events = _generate_flow_events_for_correlation_map(correlation_id_map)
    
    # Extract kernel correlation map for finding kernels
    kernel_correlation_map = {
        corr_id: data["kernels"]
        for corr_id, data in correlation_id_map.items()
    }
    
    # Process each NVTX event
    for nvtx_event in nvtx_events_list:
        nvtx_id = adapter.get_event_id(nvtx_event)
        cuda_api_events_overlapping = overlap_map.get(nvtx_id, [])
        
        if not cuda_api_events_overlapping:
            continue
        
        # Find kernels using shared function
        found_kernels = find_kernels_for_annotation(
            cuda_api_events_overlapping, kernel_correlation_map, adapter
        )
        
        # Aggregate kernel times
        time_range = aggregate_kernel_times(found_kernels, adapter)
        if time_range is None:
            continue
        
        kernel_start_time, kernel_end_time = time_range
        
        # Create nvtx-kernel event
        event = _create_nvtx_kernel_event(
            nvtx_event, kernel_start_time, kernel_end_time, device_id, options
        )
        nvtx_kernel_events.append(event)
        
        # Track this NVTX event as successfully mapped
        tid = nvtx_event.args.get("raw_tid")
        start_ns = nvtx_event.args.get("start_ns")
        nvtx_name = nvtx_event.name
        nvtx_identifier = (device_id, tid, start_ns, nvtx_name)
        mapped_nvtx_identifiers.add(nvtx_identifier)
    
    return nvtx_kernel_events, mapped_nvtx_identifiers, flow_events


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
    # Group events by device ID
    per_device_nvtx, per_device_cuda_api, per_device_kernels = _group_events_by_device(
        nvtx_events, cuda_api_events, kernel_events
    )
    
    # Get devices that have all three event types
    common_devices = (
        set(per_device_nvtx.keys()) & 
        set(per_device_cuda_api.keys()) & 
        set(per_device_kernels.keys())
    )
    
    # Create adapter for ChromeTraceEvent objects parsed from nsys SQLite
    adapter = NsysTraceEventAdapter()
    
    # Process each device
    all_nvtx_kernel_events = []
    all_mapped_nvtx_identifiers = set()
    all_flow_events = []
    
    for device_id in common_devices:
        nvtx_kernel_events, mapped_nvtx_identifiers, flow_events = _process_device_nvtx_events(
            per_device_nvtx[device_id],
            per_device_cuda_api[device_id],
            per_device_kernels[device_id],
            device_id,
            adapter,
            options,
        )
        
        all_nvtx_kernel_events.extend(nvtx_kernel_events)
        all_mapped_nvtx_identifiers.update(mapped_nvtx_identifiers)
        all_flow_events.extend(flow_events)
    
    return all_nvtx_kernel_events, all_mapped_nvtx_identifiers, all_flow_events

