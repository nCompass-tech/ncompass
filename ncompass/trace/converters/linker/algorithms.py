"""Core algorithms for linking events via correlation IDs."""

import logging
from collections import defaultdict
from typing import Any, Optional, Union

from .adapters import EventAdapter
from ..models import ChromeTraceEvent

logger = logging.getLogger(__name__)

# Event types supported by the linker algorithms
# Can be ChromeTraceEvent (from nsys SQLite) or dict[str, Any] (from Chrome trace JSON)
LinkerEvent = Union[ChromeTraceEvent, dict[str, Any]]


def find_overlapping_intervals(
    source_events: list[Any],
    target_events: list[Any],
    adapter: EventAdapter,
    source_name: str = "source",
    target_name: str = "target"
) -> dict[tuple, list[Any]]:
    """Find overlapping intervals using sweep-line algorithm.
    
    Generic implementation that works with any event format via adapter.
    
    Args:
        source_events: Events to find overlaps for (e.g., NVTX, user_annotation)
        target_events: Events to find overlaps with (e.g., CUDA API, cuda_runtime)
        adapter: Adapter to extract data from events
        source_name: Name for source event type (for logging)
        target_name: Name for target event type (for logging)
        
    Returns:
        Dictionary mapping source event ID to list of overlapping target events
    """
    # Build index map for source events
    source_index_map = {id(event): i for i, event in enumerate(source_events)}
    
    # Create mixed list of start/end events
    mixed_events = []
    source_skipped = 0
    target_skipped = 0
    
    # Add source events as start/end pairs
    for source_event in source_events:
        time_range = adapter.get_time_range(source_event)
        if time_range is None:
            source_skipped += 1
            continue
        start, end = time_range
        mixed_events.append((start, 1, source_name, source_event))
        mixed_events.append((end, -1, source_name, source_event))
    
    # Add target events as start/end pairs
    for target_event in target_events:
        time_range = adapter.get_time_range(target_event)
        if time_range is None:
            target_skipped += 1
            continue
        start, end = time_range
        mixed_events.append((start, 1, target_name, target_event))
        mixed_events.append((end, -1, target_name, target_event))
    
    # Log summary of events processed vs skipped
    if source_skipped > 0 or target_skipped > 0:
        logger.debug(
            "find_overlapping_intervals: skipped %d %s events and %d %s events without valid time ranges",
            source_skipped,
            source_name,
            target_skipped,
            target_name
        )
    
    # Sort by timestamp, then by event type (start=1 before end=-1), then by origin
    mixed_events.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Track active source intervals
    active_source_intervals = []
    result_by_index = defaultdict(list)
    
    for timestamp, event_type, event_origin, orig_event in mixed_events:
        if event_type == 1:  # Start event
            if event_origin == source_name:
                active_source_intervals.append(orig_event)
            else:  # target start
                # Add this target event to all currently active source ranges
                for source_event in active_source_intervals:
                    source_idx = source_index_map[id(source_event)]
                    result_by_index[source_idx].append(orig_event)
        else:  # End event (event_type == -1)
            if event_origin == source_name:
                active_source_intervals.remove(orig_event)
    
    # Convert to mapping by event identifier
    result = {}
    for idx, target_list in result_by_index.items():
        source_event = source_events[idx]
        event_id = adapter.get_event_id(source_event)
        result[event_id] = target_list
    
    logger.debug(
        "find_overlapping_intervals: found %d %s events with overlapping %s events",
        len(result),
        source_name,
        target_name
    )
    
    return result


def build_correlation_map(
    kernel_events: list[Any],
    adapter: EventAdapter
) -> dict[int, list[Any]]:
    """Build mapping from correlation ID to list of kernels.
    
    Args:
        kernel_events: List of kernel events
        adapter: Adapter to extract correlation ID from events
        
    Returns:
        Dictionary mapping correlation ID to list of kernel events
    """
    correlation_map = defaultdict(list)
    skipped_count = 0
    
    for kernel_event in kernel_events:
        corr_id = adapter.get_correlation_id(kernel_event)
        if corr_id is not None:
            correlation_map[corr_id].append(kernel_event)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        logger.debug(
            "build_correlation_map: skipped %d kernel events without correlationId",
            skipped_count
        )
    
    logger.debug(
        "build_correlation_map: built map with %d unique correlation IDs from %d kernels",
        len(correlation_map),
        len(kernel_events) - skipped_count
    )
    
    return correlation_map


def aggregate_kernel_times(
    kernels: list[Any],
    adapter: EventAdapter
) -> Optional[tuple[float, float]]:
    """Aggregate kernel execution times across multiple kernels.
    
    Finds the minimum start time and maximum end time across all kernels.
    
    Args:
        kernels: List of kernel events
        adapter: Adapter to extract time ranges from events
        
    Returns:
        Tuple of (min_start, max_end) or None if no valid kernels
    """
    kernel_start_time = None
    kernel_end_time = None
    
    for kernel_event in kernels:
        time_range = adapter.get_time_range(kernel_event)
        if time_range is None:
            continue
        
        kernel_start, kernel_end = time_range
        if kernel_start_time is None or kernel_start_time > kernel_start:
            kernel_start_time = kernel_start
        if kernel_end_time is None or kernel_end_time < kernel_end:
            kernel_end_time = kernel_end
    
    if kernel_start_time is None or kernel_end_time is None:
        return None
    
    return (kernel_start_time, kernel_end_time)


def find_kernels_for_annotation(
    overlapping_api_events: list[LinkerEvent],
    correlation_map: dict[int, list[LinkerEvent]],
    adapter: EventAdapter,
) -> list[LinkerEvent]:
    """Find all kernels associated with an annotation event via overlapping API events.
    
    This is a shared pattern used by both NVTX and user_annotation linkers:
    1. Extract correlation IDs from overlapping API events
    2. Look up kernels in the correlation map
    3. Collect all kernels
    
    Args:
        overlapping_api_events: List of API events (CUDA API or cuda_runtime) that overlap
        correlation_map: Mapping from correlation ID to list of kernel events
        adapter: Adapter to extract correlation IDs from events
        
    Returns:
        List of kernel events associated with the annotation event
    """
    found_kernels = []
    api_without_corr_id = 0
    api_without_kernels = 0
    
    for api_event in overlapping_api_events:
        corr_id = adapter.get_correlation_id(api_event)
        if corr_id is None:
            api_without_corr_id += 1
            continue
        if corr_id not in correlation_map:
            api_without_kernels += 1
            continue
        
        kernels = correlation_map[corr_id]
        if len(kernels) == 0:
            # API call didn't launch a kernel, skip
            api_without_kernels += 1
            continue
        
        found_kernels.extend(kernels)
    
    if api_without_corr_id > 0:
        logger.debug(
            "find_kernels_for_annotation: %d API events had no correlationId",
            api_without_corr_id
        )
    if api_without_kernels > 0:
        logger.debug(
            "find_kernels_for_annotation: %d API events had no associated kernels",
            api_without_kernels
        )
    
    return found_kernels

