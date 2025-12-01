"""Utility functions for nsys2chrome conversion."""

from typing import Any
from .models import VALID_CHROME_TRACE_PHASES


def ns_to_us(timestamp_ns: int) -> float:
    """Convert nanoseconds to microseconds.
    
    Args:
        timestamp_ns: Timestamp in nanoseconds
        
    Returns:
        Timestamp in microseconds
    """
    return timestamp_ns / 1000.0


def validate_chrome_trace(events: list[dict[str, Any]]) -> bool:
    """Validate Chrome Trace event format.
    
    Args:
        events: List of Chrome Trace events
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_fields = {"name", "ph", "ts", "pid", "tid", "cat"}
    
    for i, event in enumerate(events):
        missing = required_fields - set(event.keys())
        if missing:
            raise ValueError(
                f"Event {i} missing required fields: {missing}. "
                f"Event: {event}"
            )
        
        # Validate phase type using the shared constant
        if event["ph"] not in VALID_CHROME_TRACE_PHASES:
            raise ValueError(
                f"Event {i} has invalid phase '{event['ph']}'. "
                f"Valid phases: {sorted(VALID_CHROME_TRACE_PHASES)}"
            )
        
        # For 'X' events, duration should be present
        if event["ph"] == "X" and "dur" not in event:
            raise ValueError(f"Event {i} has phase 'X' but missing 'dur' field")
    
    return True


def write_chrome_trace(output_path: str, events: dict) -> None:
    """Write Chrome Trace events to JSON file.
    
    Args:
        output_path: Path to output JSON file
        events: List of Chrome Trace events
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(events, f)


def write_chrome_trace_gz(output_path: str, events: dict) -> None:
    """Write Chrome Trace events to gzip-compressed JSON file.
    
    Args:
        output_path: Path to output gzip-compressed JSON file (.json.gz)
        events: Chrome Trace events dictionary
    """
    import gzip
    import json
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        json.dump(events, f)

