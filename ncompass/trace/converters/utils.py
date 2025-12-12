"""Utility functions for nsys2chrome conversion."""

from typing import Any, Iterator
from .models import VALID_CHROME_TRACE_PHASES
import orjson


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


def write_chrome_trace(output_path: str, events: Iterator[dict]) -> None:
    """Write Chrome Trace events to JSON file using streaming.
    
    Args:
        output_path: Path to output JSON file
        events: Iterator of Chrome Trace event dicts
    """
    with open(output_path, 'wb') as f:
        # Write opening
        f.write(b'{"traceEvents":[')
        
        # Stream events with commas between them
        first = True
        for event in events:
            if not first:
                f.write(b',')
            else:
                first = False
            # orjson.dumps returns bytes
            f.write(orjson.dumps(event))
        
        # Write closing
        f.write(b']}')


def write_chrome_trace_gz(output_path: str, events: Iterator[dict]) -> None:
    """Write Chrome Trace events to gzip-compressed JSON file using streaming.
    
    Args:
        output_path: Path to output gzip-compressed JSON file (.json.gz)
        events: Iterator of Chrome Trace event dicts
    """
    import gzip
    
    with gzip.open(output_path, 'wb') as f:
        # Write opening
        f.write(b'{"traceEvents":[')
        
        # Stream events with commas between them
        first = True
        for event in events:
            if not first:
                f.write(b',')
            else:
                first = False
            # orjson.dumps returns bytes
            f.write(orjson.dumps(event))
        
        # Write closing
        f.write(b']}')
