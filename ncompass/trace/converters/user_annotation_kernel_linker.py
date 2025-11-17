"""Link user_annotation events to kernel events via CUDA runtime correlation.

This module implements logic to link user_annotation events (from torch.profiler)
to kernel execution times by finding overlapping CUDA runtime API calls and
using correlationId to connect them.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def _extract_event_time_range(event: dict[str, Any]) -> Optional[tuple[float, float]]:
    """Extract time range from a Chrome trace event dict.
    
    Args:
        event: Chrome trace event dictionary
        
    Returns:
        Tuple of (start_us, end_us) in microseconds, or None if invalid
    """
    if event.get("ph") != "X":
        return None
    
    ts = event.get("ts")
    dur = event.get("dur")
    
    if ts is None or dur is None:
        return None
    
    return (ts, ts + dur)


def _get_correlation_id(event: dict[str, Any]) -> Optional[int]:
    """Get correlation ID from event args.
    
    Handles both "correlation" and "correlationId" field names.
    
    Args:
        event: Chrome trace event dictionary
        
    Returns:
        Correlation ID or None
    """
    args = event.get("args", {})
    return args.get("correlation") or args.get("correlationId")


def _find_overlapping_cuda_runtime(
    user_annotation_events: list[dict[str, Any]],
    cuda_runtime_events: list[dict[str, Any]]
) -> dict[tuple, list[dict[str, Any]]]:
    """Find CUDA runtime events that overlap with each user_annotation event.
    
    Uses a sweep-line algorithm similar to nvtx_kernel_linker.
    
    Args:
        user_annotation_events: List of user_annotation event dicts
        cuda_runtime_events: List of cuda_runtime event dicts
        
    Returns:
        Dictionary mapping (name, ts, pid, tid) to list of overlapping cuda_runtime events
    """
    # Build index map for user_annotation events
    ua_index_map = {id(ua_event): i for i, ua_event in enumerate(user_annotation_events)}
    
    # Create mixed list of start/end events
    mixed_events = []
    
    # Add user_annotation events as start/end pairs
    for ua_event in user_annotation_events:
        time_range = _extract_event_time_range(ua_event)
        if time_range is None:
            continue
        start, end = time_range
        mixed_events.append((start, 1, "user_annotation", ua_event))
        mixed_events.append((end, -1, "user_annotation", ua_event))
    
    # Add cuda_runtime events as start/end pairs
    for cuda_runtime_event in cuda_runtime_events:
        time_range = _extract_event_time_range(cuda_runtime_event)
        if time_range is None:
            continue
        start, end = time_range
        mixed_events.append((start, 1, "cuda_runtime", cuda_runtime_event))
        mixed_events.append((end, -1, "cuda_runtime", cuda_runtime_event))
    
    # Sort by timestamp, then by event type (start=1 before end=-1), then by origin
    mixed_events.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Track active user_annotation intervals
    active_ua_intervals = []
    result_by_index = defaultdict(list)
    
    for timestamp, event_type, event_origin, orig_event in mixed_events:
        if event_type == 1:  # Start event
            if event_origin == "user_annotation":
                active_ua_intervals.append(orig_event)
            else:  # cuda_runtime start
                # Add this CUDA runtime call to all currently active user_annotation ranges
                for ua_event in active_ua_intervals:
                    ua_idx = ua_index_map[id(ua_event)]
                    result_by_index[ua_idx].append(orig_event)
        else:  # End event (event_type == -1)
            if event_origin == "user_annotation":
                active_ua_intervals.remove(orig_event)
    
    # Convert to mapping by event identifier
    result = {}
    for idx, cuda_runtime_list in result_by_index.items():
        ua_event = user_annotation_events[idx]
        # Use a tuple that uniquely identifies the event
        event_id = (
            ua_event.get("name", ""),
            ua_event.get("ts"),
            ua_event.get("pid"),
            ua_event.get("tid")
        )
        result[event_id] = cuda_runtime_list
    
    return result


def _load_chrome_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    """Load Chrome trace JSON file.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        List of trace events
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)
    
    # Handle both array format and object with traceEvents key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    else:
        raise ValueError(f"Unexpected trace format in {trace_path}")


def link_user_annotation_to_kernels(
    trace_path: str | Path,
) -> list[dict[str, Any]]:
    """Link user_annotation events to kernel events via CUDA runtime correlation.
    
    This function loads a Chrome trace file and:
    1. Finds overlapping intervals between user_annotation and CUDA runtime events
    2. Uses correlationId to link CUDA runtime calls to kernels
    3. Creates new "gpu_user_annotation" events that span kernel execution times
    
    Replacement logic:
    - If both gpu_user_annotation and user_annotation exist (same name) → generates replacement
    - If only user_annotation exists → generates new gpu_user_annotation event
    - If only gpu_user_annotation exists → no replacement (leaves it as is)
    
    Uses pid/tid from existing gpu_user_annotation (if exists) or from kernels.
    
    Args:
        trace_path: Path to Chrome trace JSON file
        
    Returns:
        List of new gpu_user_annotation event dicts (category="gpu_user_annotation")
    """
    # Load trace file
    trace_events = _load_chrome_trace(trace_path)
    
    # Separate events by category
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
    
    linked_events = []
    
    # Early return if no user_annotation events or missing required event types
    if not user_annotation_events:
        return []
    
    if not cuda_runtime_events or not kernel_events:
        return []
    
    # Build mapping of gpu_user_annotation events by name
    # Used to determine pid/tid for replacement events
    gpu_ua_by_name = defaultdict(list)
    if gpu_user_annotation_events:
        for gpu_ua_event in gpu_user_annotation_events:
            name = gpu_ua_event.get("name", "")
            if name:
                gpu_ua_by_name[name].append(gpu_ua_event)
    
    # Find overlapping CUDA runtime events for each user_annotation event
    overlap_map = _find_overlapping_cuda_runtime(user_annotation_events, cuda_runtime_events)
    
    # Build correlationId -> kernels[] mapping
    # Multiple kernels can share the same correlationId (e.g., CUDA graphs)
    correlation_id_map = defaultdict(list)
    
    for kernel_event in kernel_events:
        corr_id = _get_correlation_id(kernel_event)
        if corr_id is not None:
            correlation_id_map[corr_id].append(kernel_event)
    
    # For each user_annotation event with overlapping CUDA runtime calls, find kernels
    for ua_event in user_annotation_events:
        ua_name = ua_event.get("name", "")
        
        # Logic:
        # - If both gpu_user_annotation and user_annotation exist → generate replacement
        # - If only user_annotation exists → generate new event
        # - If only gpu_user_annotation exists → skip (handled by caller not passing it here)
        event_id = (
            ua_event.get("name", ""),
            ua_event.get("ts"),
            ua_event.get("pid"),
            ua_event.get("tid")
        )
        
        overlapping_cuda_runtime = overlap_map.get(event_id, [])
        
        if not overlapping_cuda_runtime:
            continue
        
        kernel_start_time = None
        kernel_end_time = None
        found_kernels = []
        
        # Find kernels via correlationId from overlapping CUDA runtime calls
        for cuda_runtime_event in overlapping_cuda_runtime:
            corr_id = _get_correlation_id(cuda_runtime_event)
            if corr_id is None or corr_id not in correlation_id_map:
                continue
            
            kernels = correlation_id_map[corr_id]
            if len(kernels) == 0:
                # CUDA runtime call didn't launch a kernel, skip
                continue
            
            # Process ALL kernels with this correlationId
            for kernel_event in kernels:
                found_kernels.append(kernel_event)
                time_range = _extract_event_time_range(kernel_event)
                if time_range is None:
                    continue
                
                kernel_start, kernel_end = time_range
                # Track min start and max end times across ALL kernels
                if kernel_start_time is None or kernel_start_time > kernel_start:
                    kernel_start_time = kernel_start
                if kernel_end_time is None or kernel_end_time < kernel_end:
                    kernel_end_time = kernel_end
        
        # Create user_annotation-kernel event if we found kernels
        if kernel_start_time is not None and kernel_end_time is not None:
            duration = kernel_end_time - kernel_start_time
            
            # Determine pid and tid:
            # - If gpu_user_annotation exists for this name → use its pid/tid
            # - Otherwise → use pid/tid from kernels
            device_pid = None
            device_tid = None
            
            # Check if there's an existing gpu_user_annotation for this name
            if ua_name in gpu_ua_by_name:
                # Use pid/tid from the existing gpu_user_annotation
                gpu_ua = gpu_ua_by_name[ua_name][0]
                device_pid = gpu_ua.get("pid")
                device_tid = gpu_ua.get("tid")
            else:
                # Get pid/tid from kernels (use first kernel's pid/tid)
                for kernel_event in found_kernels:
                    pid = kernel_event.get("pid")
                    tid = kernel_event.get("tid")
                    
                    # For PyTorch profiler format: pid is numeric, get device from args
                    if isinstance(pid, (int, float)) or pid is None:
                        device_id = kernel_event.get("args", {}).get("device")
                        if device_id is not None:
                            device_pid = device_id
                            device_tid = tid
                            break
                        elif pid is not None:
                            device_pid = pid
                            device_tid = tid
                            break
                    # For nsys2chrome format: pid is "Device X" string
                    elif isinstance(pid, str) and pid.startswith("Device "):
                        device_pid = pid
                        device_tid = tid
                        break
            
            # Fallback if still not found
            if device_pid is None:
                # Default to numeric format (PyTorch profiler style)
                device_pid = 0
                device_tid = 0
            
            # Use "gpu_user_annotation" category to match existing format
            new_event = {
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
            
            linked_events.append(new_event)
    
    return linked_events

