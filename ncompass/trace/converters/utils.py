"""Utility functions for nsys2chrome conversion."""

from typing import Any, Iterator
from .models import VALID_CHROME_TRACE_PHASES
import orjson

# Unicode arrow for overflow tracks (U+21B3)
_OVERFLOW_PREFIX = "↳ "


def _process_event_for_overlap(
    event: dict,
    max_end: dict[tuple, float],
) -> dict:
    """
    Process a single event for overlap detection and assign to virtual track if needed.
    
    Perfetto requires strict nesting for events on the same track. Events that partially
    overlap (start during previous but end after) get dropped. This function detects
    such events and moves them to a virtual overflow track.
    
    Args:
        event: Chrome Trace event dict (must have ph, ts, dur, pid, tid for X events)
        max_end: Dict mapping (pid, tid) -> max end time seen so far. Modified in place.
        
    Returns:
        The event, potentially with modified tid if moved to overflow track.
    """
    # Only process Complete events (phase X) with duration
    if event.get('ph') != 'X' or 'ts' not in event or 'dur' not in event:
        return event
    
    pid = event.get('pid')
    original_tid = event.get('tid')
    ts = event['ts']
    dur = event['dur']
    event_end = ts + dur
    
    original_key = (pid, original_tid)
    overflow_tid = f"{_OVERFLOW_PREFIX}{original_tid}"
    overflow_key = (pid, overflow_tid)
    
    orig_max = max_end.get(original_key, float('-inf'))
    
    # Check if event fits on original track:
    # - No overlap (starts after previous ends): ts >= orig_max
    # - Fully nested (ends before previous ends): event_end <= orig_max
    if ts >= orig_max or event_end <= orig_max:
        # Keep on original track
        max_end[original_key] = max(orig_max, event_end)
        return event
    else:
        # Partial overlap - move to overflow track
        # Make a copy to avoid mutating the original
        event = dict(event)
        event['tid'] = overflow_tid
        overflow_max = max_end.get(overflow_key, float('-inf'))
        max_end[overflow_key] = max(overflow_max, event_end)
        return event


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
    
    Automatically handles overlapping events by moving them to virtual overflow
    tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    
    Args:
        output_path: Path to output JSON file
        events: Iterator of Chrome Trace event dicts (must be sorted by timestamp)
    """
    # Track max end time per (pid, tid) for overlap detection
    max_end: dict[tuple, float] = {}
    
    with open(output_path, 'wb') as f:
        # Write opening with newline
        f.write(b'{"traceEvents":[\n')
        
        # Stream events with commas between them
        # Each event on its own line to avoid Perfetto parser issues with very long lines
        first = True
        for event in events:
            # Process event for overlap and potentially assign to overflow track
            event = _process_event_for_overlap(event, max_end)
            
            if not first:
                f.write(b',\n')
            else:
                first = False
            # orjson.dumps returns bytes
            f.write(orjson.dumps(event))
        
        # Write closing with newline
        f.write(b'\n]}')


def write_chrome_trace_gz(output_path: str, events: Iterator[dict]) -> None:
    """Write Chrome Trace events to gzip-compressed JSON file using streaming.
    
    Automatically handles overlapping events by moving them to virtual overflow
    tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    
    Args:
        output_path: Path to output gzip-compressed JSON file (.json.gz)
        events: Iterator of Chrome Trace event dicts (must be sorted by timestamp)
    """
    import gzip
    
    # Track max end time per (pid, tid) for overlap detection
    max_end: dict[tuple, float] = {}
    
    with gzip.open(output_path, 'wb') as f:
        # Write opening with newline
        f.write(b'{"traceEvents":[\n')
        
        # Stream events with commas between them
        # Each event on its own line to avoid Perfetto parser issues with very long lines
        first = True
        for event in events:
            # Process event for overlap and potentially assign to overflow track
            event = _process_event_for_overlap(event, max_end)
            
            if not first:
                f.write(b',\n')
            else:
                first = False
            # orjson.dumps returns bytes
            f.write(orjson.dumps(event))
        
        # Write closing with newline
        f.write(b'\n]}')
