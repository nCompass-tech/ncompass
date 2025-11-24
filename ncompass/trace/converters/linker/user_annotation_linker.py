"""Link user_annotation events to kernel events via CUDA runtime correlation.

This module uses shared algorithms from linker.algorithms to link user_annotation
events (from torch.profiler) to kernel execution times by finding overlapping
CUDA runtime API calls and using correlationId to connect them.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from ncompass.trace.infra.utils import logger
from .algorithms import (
    find_overlapping_intervals,
    build_correlation_map,
    aggregate_kernel_times,
    find_kernels_for_annotation,
)
from .adapters import ChromeTraceEventAdapter


def _load_chrome_trace(trace_path: str | Path) -> dict[str, Any]:
    """Load Chrome trace JSON file.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        Dict with 'traceEvents' key
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, dict) or "traceEvents" not in data:
        raise ValueError(f"Expected dict with 'traceEvents' key in {trace_path}")
    
    return data


def _filter_events_by_category(trace_events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate trace events by category.
    
    Args:
        trace_events: All trace events
        
    Returns:
        Tuple of (user_annotation_events, gpu_user_annotation_events, cuda_runtime_events, kernel_events)
    """
    user_annotation_events = [
        e for e in trace_events 
        if e.get("cat") == "user_annotation" and e.get("ph") == "X"
    ]
    
    gpu_user_annotation_events = [
        e for e in trace_events 
        if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
    ]
    
    cuda_runtime_events = [
        e for e in trace_events 
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"
    ]
    
    kernel_events = [
        e for e in trace_events 
        if e.get("cat") == "kernel" and e.get("ph") == "X"
    ]
    
    return user_annotation_events, gpu_user_annotation_events, cuda_runtime_events, kernel_events


def _determine_device_pid_tid(
    ua_name: str,
    found_kernels: list[dict[str, Any]],
    gpu_ua_by_name: dict[str, list[dict[str, Any]]],
) -> tuple[Any, Any]:
    """Determine pid and tid for a gpu_user_annotation event.
    
    Priority:
    1. If gpu_user_annotation exists for this name → use its pid/tid
    2. Otherwise → use pid/tid from kernels (handles PyTorch and nsys formats)
    3. Fallback → default to (0, 0)
    
    Args:
        ua_name: Name of the user_annotation event
        found_kernels: List of kernel events found for this annotation
        gpu_ua_by_name: Mapping of name to existing gpu_user_annotation events
        
    Returns:
        Tuple of (device_pid, device_tid)
    """
    # Check if there's an existing gpu_user_annotation for this name
    if ua_name in gpu_ua_by_name:
        # Use pid/tid from the existing gpu_user_annotation
        gpu_ua = gpu_ua_by_name[ua_name][0]
        return gpu_ua.get("pid"), gpu_ua.get("tid")
    
    # Get pid/tid from kernels (use first kernel's pid/tid)
    for kernel_event in found_kernels:
        pid = kernel_event.get("pid")
        tid = kernel_event.get("tid")
        
        # For PyTorch profiler format: pid is numeric, get device from args
        if isinstance(pid, (int, float)) or pid is None:
            device_id = kernel_event.get("args", {}).get("device")
            if device_id is not None:
                return device_id, tid
            elif pid is not None:
                return pid, tid
        # For nsys2chrome format: pid is "Device X" string
        elif isinstance(pid, str) and pid.startswith("Device "):
            return pid, tid
    
    # Fallback: default to numeric format (PyTorch profiler style)
    return 0, 0


def _create_gpu_user_annotation_event(
    ua_event: dict[str, Any],
    kernel_start_time: float,
    kernel_end_time: float,
    found_kernels: list[dict[str, Any]],
    device_pid: Any,
    device_tid: Any,
) -> dict[str, Any]:
    """Create a single gpu_user_annotation event.
    
    Args:
        ua_event: The user_annotation event
        kernel_start_time: Start time of aggregated kernels (microseconds)
        kernel_end_time: End time of aggregated kernels (microseconds)
        found_kernels: List of kernel events found
        device_pid: PID for the GPU device
        device_tid: TID for the GPU device
        
    Returns:
        Dictionary representing the gpu_user_annotation event
    """
    duration = kernel_end_time - kernel_start_time
    
    return {
        "name": ua_event.get("name", ""),
        "ph": "X",
        "cat": "gpu_user_annotation",
        "ts": kernel_start_time,
        "dur": duration,
        "pid": device_pid,
        "tid": device_tid,
        "args": {
            "original_ts": ua_event.get("ts"),
            "original_dur": ua_event.get("dur"),
            "kernel_count": len(found_kernels),
            "original_pid": ua_event.get("pid"),
            "original_tid": ua_event.get("tid"),
        }
    }


def _filter_and_replace_events(
    trace_events: list[dict[str, Any]],
    new_events: list[dict[str, Any]],
    gpu_user_annotation_events: list[dict[str, Any]],
    user_annotation_events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Filter out events being replaced and combine with new events.
    
    Replacement logic:
    - If both gpu_user_annotation and user_annotation exist (same name) → remove gpu_user_annotation
    - Always keep user_annotation events
    
    Args:
        trace_events: All original trace events
        new_events: Newly created gpu_user_annotation events
        gpu_user_annotation_events: Existing gpu_user_annotation events
        user_annotation_events: Existing user_annotation events
        
    Returns:
        Tuple of (filtered_events, removed_gpu_ua_count)
    """
    # Build sets of names that will be replaced
    replaced_names = {e.get("name", "") for e in new_events if e.get("name")}
    
    # Build sets to determine which events to remove
    gpu_ua_names = {e.get("name", "") for e in gpu_user_annotation_events if e.get("name")}
    ua_names = {e.get("name", "") for e in user_annotation_events if e.get("name")}
    
    both_exist_names = gpu_ua_names & ua_names & replaced_names
    
    # Filter replaced events
    filtered_events = []
    removed_gpu_ua_count = 0
    
    for event in trace_events:
        cat = event.get("cat", "")
        name = event.get("name", "")
        
        # Remove gpu_user_annotation if both exist (being replaced)
        if cat == "gpu_user_annotation" and name in both_exist_names:
            removed_gpu_ua_count += 1
            continue
        
        # Keep user_annotation events in all cases
        filtered_events.append(event)
    
    # Combine filtered events with new events
    linked_events = filtered_events + new_events
    
    return linked_events, removed_gpu_ua_count


def _log_linking_statistics(
    trace_events: list[dict[str, Any]],
    new_events: list[dict[str, Any]],
    removed_gpu_ua_count: int,
    gpu_user_annotation_events: list[dict[str, Any]],
    user_annotation_events: list[dict[str, Any]],
) -> None:
    """Log verbose statistics about the linking process.
    
    Args:
        trace_events: All original trace events
        new_events: Newly created gpu_user_annotation events
        removed_gpu_ua_count: Number of removed gpu_user_annotation events
        gpu_user_annotation_events: Existing gpu_user_annotation events
        user_annotation_events: Existing user_annotation events
    """
    # Count events for statistics
    counts = {
        "user_annotation": len([e for e in trace_events if e.get("cat") == "user_annotation" and e.get("ph") == "X"]),
        "gpu_user_annotation": len([e for e in trace_events if e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"]),
        "cuda_runtime": len([e for e in trace_events if e.get("cat") == "cuda_runtime" and e.get("ph") == "X"]),
        "kernel": len([e for e in trace_events if e.get("cat") == "kernel" and e.get("ph") == "X"]),
    }
    
    logger.info(f"Found {counts['user_annotation']} user_annotation events")
    logger.info(f"Found {counts['gpu_user_annotation']} gpu_user_annotation events")
    logger.info(f"Found {counts['cuda_runtime']} cuda_runtime events")
    logger.info(f"Found {counts['kernel']} kernel events")
    
    if removed_gpu_ua_count > 0:
        logger.info(f"\nRemoved {removed_gpu_ua_count} old gpu_user_annotation events (replaced)")
    
    logger.info(f"Linked {len(new_events)} user_annotation events to kernels")
    
    if new_events:
        # Build replacement type sets
        replaced_names = {e.get("name", "") for e in new_events if e.get("name")}
        gpu_ua_names = {e.get("name", "") for e in gpu_user_annotation_events if e.get("name")}
        ua_names = {e.get("name", "") for e in user_annotation_events if e.get("name")}
        both_exist_names = gpu_ua_names & ua_names & replaced_names
        ua_only_names = (ua_names - gpu_ua_names) & replaced_names
        
        logger.info("\nNew/replaced gpu_user_annotation events:")
        for event in new_events:
            original_dur = event["args"].get("original_dur", 0)
            new_dur = event["dur"]
            kernel_count = event["args"].get("kernel_count", 0)
            event_name = event["name"]
            
            if event_name in both_exist_names:
                replacement_type = "replaced (both existed, user_annotation kept)"
            elif event_name in ua_only_names:
                replacement_type = "new (user_annotation kept)"
            else:
                replacement_type = "new"
            
            logger.info(
                f"  '{event_name}' ({replacement_type}): "
                f"{original_dur:.2f} -> {new_dur:.2f} us "
                f"({kernel_count} kernels, pid={event['pid']}, tid={event['tid']})"
            )


def link_user_annotation_to_kernels(
    trace_path: str | Path,
    verbose: bool = False,
) -> dict[str, Any]:
    """Link user_annotation events to kernel events via CUDA runtime correlation.
    
    This function loads a Chrome trace file and:
    1. Finds overlapping intervals between user_annotation and CUDA runtime events
    2. Uses correlationId to link CUDA runtime calls to kernels
    3. Creates new "gpu_user_annotation" events that span kernel execution times
    4. Filters out old events being replaced and returns complete linked events list
    
    Replacement logic:
    - If both gpu_user_annotation and user_annotation exist (same name) → generates replacement (removes gpu_user_annotation, keeps user_annotation)
    - If only user_annotation exists → generates new gpu_user_annotation event (keeps user_annotation)
    - If only gpu_user_annotation exists → no replacement (leaves it as is)
    
    Uses pid/tid from existing gpu_user_annotation (if exists) or from kernels.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        verbose: If True, print detailed statistics about linking process
        
    Returns:
        Complete trace dict with linked gpu_user_annotation events in 'traceEvents',
        ready to be written to a file
    """
    # Load trace file
    trace_data = _load_chrome_trace(trace_path)
    trace_events = trace_data['traceEvents']
    
    # Separate events by category
    user_annotation_events, gpu_user_annotation_events, cuda_runtime_events, kernel_events = _filter_events_by_category(trace_events)
    
    # Early return if no user_annotation events or missing required event types
    if not user_annotation_events:
        if verbose:
            logger.info("No user_annotation events found in trace")
        return {**trace_data, 'traceEvents': trace_events}
    
    if not cuda_runtime_events or not kernel_events:
        if verbose:
            logger.info("Missing required event types (cuda_runtime or kernel events)")
        return {**trace_data, 'traceEvents': trace_events}
    
    # Build mapping of gpu_user_annotation events by name
    gpu_ua_by_name = defaultdict(list)
    if gpu_user_annotation_events:
        for gpu_ua_event in gpu_user_annotation_events:
            name = gpu_ua_event.get("name", "")
            if name:
                gpu_ua_by_name[name].append(gpu_ua_event)
    
    # Use shared algorithms with Chrome trace JSON adapter
    adapter = ChromeTraceEventAdapter()
    
    # Find overlapping CUDA runtime events for each user_annotation event
    overlap_map = find_overlapping_intervals(
        user_annotation_events, cuda_runtime_events, adapter,
        "user_annotation", "cuda_runtime"
    )
    
    # Build correlationId -> kernels[] mapping
    correlation_id_map = build_correlation_map(kernel_events, adapter)
    
    # Process each user_annotation event
    new_events = []
    for ua_event in user_annotation_events:
        ua_name = ua_event.get("name", "")
        event_id = adapter.get_event_id(ua_event)
        overlapping_cuda_runtime = overlap_map.get(event_id, [])
        
        if not overlapping_cuda_runtime:
            continue
        
        # Find kernels using shared function
        found_kernels = find_kernels_for_annotation(
            overlapping_cuda_runtime, correlation_id_map, adapter
        )
        
        # Aggregate kernel times
        time_range = aggregate_kernel_times(found_kernels, adapter)
        if time_range is None:
            continue
        
        kernel_start_time, kernel_end_time = time_range
        
        # Determine pid/tid
        device_pid, device_tid = _determine_device_pid_tid(
            ua_name, found_kernels, gpu_ua_by_name
        )
        
        # Create new event
        new_event = _create_gpu_user_annotation_event(
            ua_event, kernel_start_time, kernel_end_time,
            found_kernels, device_pid, device_tid
        )
        new_events.append(new_event)
    
    # If no new events were created, return original trace unchanged
    if not new_events:
        if verbose:
            logger.info("No new linked events created")
        return {**trace_data, 'traceEvents': trace_events}
    
    # Filter and replace events
    linked_events, removed_gpu_ua_count = _filter_and_replace_events(
        trace_events, new_events, gpu_user_annotation_events, user_annotation_events
    )
    
    # Log statistics if verbose
    if verbose:
        _log_linking_statistics(
            trace_events, new_events, removed_gpu_ua_count,
            gpu_user_annotation_events, user_annotation_events
        )
    
    return {**trace_data, 'traceEvents': linked_events}

