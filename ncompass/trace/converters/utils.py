"""Utility functions for nsys2chrome conversion."""

import json
from typing import Any, Iterator, Union
from .models import VALID_CHROME_TRACE_PHASES, ChromeTraceEvent


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
        events: Dict with traceEvents key containing event list
    """
    with open(output_path, 'w') as f:
        json.dump(events, f)


class StreamingChromeTraceWriter:
    """Streaming writer for Chrome Trace JSON format.
    
    Writes events incrementally to avoid loading all events into memory.
    The output format is: {"traceEvents": [event1, event2, ...]}
    
    Usage:
        with StreamingChromeTraceWriter("output.json") as writer:
            for event in event_generator():
                writer.write_event(event)
    """
    
    def __init__(self, output_path: str):
        """Initialize the streaming writer.
        
        Args:
            output_path: Path to the output JSON file
        """
        self.output_path = output_path
        self._file = None
        self._first_event = True
    
    def __enter__(self) -> "StreamingChromeTraceWriter":
        """Open file and write JSON header."""
        self._file = open(self.output_path, 'w')
        self._file.write('{"traceEvents":[')
        self._first_event = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Write JSON footer and close file."""
        if self._file:
            self._file.write(']}')
            self._file.close()
            self._file = None
    
    def write_event(self, event: Union[ChromeTraceEvent, dict[str, Any]]) -> None:
        """Write a single event to the file.
        
        Args:
            event: ChromeTraceEvent or dict to write
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use as context manager.")
        
        # Convert to dict if needed
        if isinstance(event, ChromeTraceEvent):
            event_dict = event.to_dict()
        else:
            event_dict = event
        
        # Handle comma separator
        if self._first_event:
            self._first_event = False
        else:
            self._file.write(',')
        
        # Write the event JSON (compact format)
        json.dump(event_dict, self._file, separators=(',', ':'))
    
    def write_events(self, events: Iterator[Union[ChromeTraceEvent, dict[str, Any]]]) -> int:
        """Write multiple events from an iterator.
        
        Args:
            events: Iterator of events to write
            
        Returns:
            Number of events written
        """
        count = 0
        for event in events:
            self.write_event(event)
            count += 1
        return count

