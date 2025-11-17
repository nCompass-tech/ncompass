"""Utility functions for nsys2chrome conversion."""

from typing import Any


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
        
        # Validate phase type
        valid_phases = {"X", "B", "E", "i", "n", "O", "C", "b", "e", "s", "t", "f", "P", "N", "M", "c"}
        if event["ph"] not in valid_phases:
            raise ValueError(
                f"Event {i} has invalid phase '{event['ph']}'. "
                f"Valid phases: {valid_phases}"
            )
        
        # For 'X' events, duration should be present
        if event["ph"] == "X" and "dur" not in event:
            raise ValueError(f"Event {i} has phase 'X' but missing 'dur' field")
    
    return True


def write_chrome_trace(output_path: str, events: list[dict[str, Any]]) -> None:
    """Write Chrome Trace events to JSON file.
    
    Args:
        output_path: Path to output JSON file
        events: List of Chrome Trace events
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(events, f)

