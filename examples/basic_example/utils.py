"""Utility functions for trace processing and statistics printing."""

from typing import Any
from ncompass.trace.infra.utils import logger


def count_events_by_category(trace_events: list[dict[str, Any]]) -> dict[str, int]:
    """Count events by category.
    
    Args:
        trace_events: List of trace event dictionaries
        
    Returns:
        Dictionary with counts for: user_annotation, gpu_user_annotation, 
        cuda_runtime, kernel
    """
    counts = {
        "user_annotation": 0,
        "gpu_user_annotation": 0,
        "cuda_runtime": 0,
        "kernel": 0,
    }
    
    for event in trace_events:
        cat = event.get("cat", "")
        ph = event.get("ph", "")
        
        if ph != "X":
            continue
        
        if cat == "user_annotation":
            counts["user_annotation"] += 1
        elif cat == "gpu_user_annotation":
            counts["gpu_user_annotation"] += 1
        elif cat == "cuda_runtime":
            counts["cuda_runtime"] += 1
        elif cat == "kernel":
            counts["kernel"] += 1
    
    return counts


def print_event_statistics(counts: dict[str, int], use_logger: bool = False) -> None:
    """Print event counts.
    
    Args:
        counts: Dictionary with event counts from count_events_by_category
        use_logger: If True, use logger.info; otherwise use print
    """
    output_func = logger.info if use_logger else print
    
    output_func(f"Found {counts['user_annotation']} user_annotation events")
    output_func(f"Found {counts['gpu_user_annotation']} gpu_user_annotation events")
    output_func(f"Found {counts['cuda_runtime']} cuda_runtime events")
    output_func(f"Found {counts['kernel']} kernel events")


def calculate_replacement_sets(
    new_events: list[dict[str, Any]],
    trace_events: list[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    """Calculate replacement sets for event filtering.
    
    Args:
        new_events: List of new linked events
        trace_events: List of original trace events
        
    Returns:
        Tuple of (both_exist_names, ua_only_names) sets
    """
    # Build sets of names that will be replaced
    replaced_names = {e.get("name", "") for e in new_events if e.get("name")}
    
    # Separate events by category for replacement logic
    gpu_user_annotation_events = [
        e for e in trace_events 
        if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
    ]
    user_annotation_events = [
        e for e in trace_events 
        if e.get("cat") == "user_annotation" and e.get("ph") == "X"
    ]
    
    # Build sets to determine which events to remove
    gpu_ua_names = {e.get("name", "") for e in gpu_user_annotation_events if e.get("name")}
    ua_names = {e.get("name", "") for e in user_annotation_events if e.get("name")}
    
    both_exist_names = gpu_ua_names & ua_names & replaced_names
    ua_only_names = (ua_names - gpu_ua_names) & replaced_names
    
    return both_exist_names, ua_only_names


def filter_replaced_events(
    trace_events: list[dict[str, Any]],
    both_exist_names: set[str],
    ua_only_names: set[str]
) -> tuple[list[dict[str, Any]], int, int]:
    """Filter out events that are being replaced.
    
    Args:
        trace_events: List of trace events
        both_exist_names: Set of names where both gpu_user_annotation and user_annotation exist
        ua_only_names: Set of names where only user_annotation exists
        
    Returns:
        Tuple of (filtered_events, removed_gpu_ua_count, removed_ua_count)
    """
    filtered_events = []
    removed_gpu_ua_count = 0
    removed_ua_count = 0
    
    for event in trace_events:
        cat = event.get("cat", "")
        name = event.get("name", "")
        
        # Remove gpu_user_annotation if both exist (being replaced)
        if cat == "gpu_user_annotation" and name in both_exist_names:
            removed_gpu_ua_count += 1
            continue
        
        # Remove user_annotation if only user_annotation exists (being replaced)
        if cat == "user_annotation" and name in ua_only_names:
            removed_ua_count += 1
            continue
        
        filtered_events.append(event)
    
    return filtered_events, removed_gpu_ua_count, removed_ua_count


def print_replacement_statistics(
    removed_gpu_ua_count: int,
    removed_ua_count: int,
    new_events: list[dict[str, Any]],
    both_exist_names: set[str],
    ua_only_names: set[str],
    use_logger: bool = False
) -> None:
    """Print replacement statistics (always verbose).
    
    Args:
        removed_gpu_ua_count: Number of removed gpu_user_annotation events
        removed_ua_count: Number of removed user_annotation events
        new_events: List of new linked events
        both_exist_names: Set of names where both existed
        ua_only_names: Set of names where only user_annotation existed
        use_logger: If True, use logger.info; otherwise use print
    """
    output_func = logger.info if use_logger else print
    
    if removed_gpu_ua_count > 0:
        output_func(f"\nRemoved {removed_gpu_ua_count} old gpu_user_annotation events (replaced)")
    if removed_ua_count > 0:
        output_func(f"Removed {removed_ua_count} old user_annotation events (replaced)")
    
    if new_events:
        output_func("\nNew/replaced gpu_user_annotation events:")
        for event in new_events:
            original_dur = event["args"].get("original_dur", 0)
            new_dur = event["dur"]
            kernel_count = event["args"].get("kernel_count", 0)
            event_name = event["name"]
            
            if event_name in both_exist_names:
                replacement_type = "replaced (both existed)"
            elif event_name in ua_only_names:
                replacement_type = "replaced (user_annotation only)"
            else:
                replacement_type = "new"
            
            output_func(
                f"  '{event_name}' ({replacement_type}): "
                f"{original_dur:.2f} -> {new_dur:.2f} us "
                f"({kernel_count} kernels, pid={event['pid']}, tid={event['tid']})"
            )

