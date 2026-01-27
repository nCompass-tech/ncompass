"""Utility functions for nsys2chrome conversion."""

import gzip
import shutil
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import orjson

from .models import VALID_CHROME_TRACE_PHASES
from ncompass.types import NCBase

_OVERFLOW_NAME_PREFIX = "↳ "
_OVERFLOW_TID_OFFSET = 100


def _compute_overflow_tid(original_tid):
    if isinstance(original_tid, int):
        return original_tid + _OVERFLOW_TID_OFFSET
    try:
        return int(original_tid) + _OVERFLOW_TID_OFFSET
    except (ValueError, TypeError):
        return f"{_OVERFLOW_NAME_PREFIX}{original_tid}"


class OverflowState(NCBase):
    def __init__(self):
        self.max_end: dict[tuple, float] = {}
        self.thread_names: dict[tuple, str] = {}
        self.thread_sort_indices: dict[tuple, int] = {}
        self.overflow_tracks: dict[tuple, tuple] = {}
        self.moved_events: dict[tuple, int] = {}

    def extract_thread_metadata(self, event: dict) -> None:
        if event.get('ph') != 'M':
            return
        key = (event.get('pid'), event.get('tid'))
        name = event.get('name')
        args = event.get('args', {})
        if name == 'thread_name':
            if thread_name := args.get('name'):
                self.thread_names[key] = thread_name
        elif name == 'thread_sort_index':
            if sort_index := args.get('sort_index'):
                self.thread_sort_indices[key] = sort_index

    def generate_overflow_metadata(self) -> list[dict]:
        events = []
        for (pid, overflow_tid), (_, original_tid) in self.overflow_tracks.items():
            original_key = (pid, original_tid)

            original_name = self.thread_names.get(original_key)
            overflow_name = f"{_OVERFLOW_NAME_PREFIX}{original_name}" if original_name else f"{_OVERFLOW_NAME_PREFIX}{original_tid}"

            events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": pid,
                "tid": overflow_tid,
                "ts": 0.0,
                "args": {"name": overflow_name}
            })

            if original_key in self.thread_sort_indices:
                events.append({
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": overflow_tid,
                    "ts": 0.0,
                    "args": {"sort_index": self.thread_sort_indices[original_key] + 1}
                })
        return events


def _process_event_for_overlap(event: dict, state: OverflowState) -> dict:
    if event.get('ph') != 'X' or 'ts' not in event or 'dur' not in event:
        return event
    
    cat = event.get('cat', '')
    if 'user_annotation' in cat:
        return event
    
    pid = event.get('pid')
    original_tid = event.get('tid')
    ts = event['ts']
    dur = event['dur']
    event_end = ts + dur
    
    original_key = (pid, original_tid)
    overflow_tid = _compute_overflow_tid(original_tid)
    overflow_key = (pid, overflow_tid)
    
    orig_max = state.max_end.get(original_key, float('-inf'))
    
    if ts >= orig_max or event_end <= orig_max:
        state.max_end[original_key] = max(orig_max, event_end)
        return event
    else:
        event = dict(event)
        event['tid'] = overflow_tid
        overflow_max = state.max_end.get(overflow_key, float('-inf'))
        state.max_end[overflow_key] = max(overflow_max, event_end)
        
        state.overflow_tracks[overflow_key] = original_key
        state.moved_events[(pid, original_tid, ts)] = overflow_tid
        return event


def _update_flow_event_if_needed(event: dict, state: OverflowState) -> dict:
    ph = event.get('ph')
    if ph not in ('s', 'f'):
        return event
    key = (event.get('pid'), event.get('tid'), event.get('ts'))
    if new_tid := state.moved_events.get(key):
        event = dict(event)
        event['tid'] = new_tid
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
    event_list = list(events)
    state = OverflowState()
    
    for event in event_list:
        state.extract_thread_metadata(event)
    
    with open(output_path, 'wb') as f:
        f.write(b'{"traceEvents":[\n')
        
        first = True
        for event in event_list:
            event = _process_event_for_overlap(event, state)
            event = _update_flow_event_if_needed(event, state)
            
            if not first:
                f.write(b',\n')
            else:
                first = False
            f.write(orjson.dumps(event))
        
        for metadata_event in state.generate_overflow_metadata():
            f.write(b',\n')
            f.write(orjson.dumps(metadata_event))
        
        f.write(b'\n]}')


def write_chrome_trace_gz(output_path: str, events: Iterator[dict]) -> None:
    """Write Chrome Trace events to gzip-compressed JSON file using streaming.
    
    Automatically handles overlapping events by moving them to virtual overflow
    tracks (e.g., "↳ Stream 7") to prevent Perfetto from dropping them.
    
    Args:
        output_path: Path to output gzip-compressed JSON file (.json.gz)
        events: Iterator of Chrome Trace event dicts (must be sorted by timestamp)
    """
    event_list = list(events)
    state = OverflowState()
    
    for event in event_list:
        state.extract_thread_metadata(event)
    
    with gzip.open(output_path, 'wb') as f:
        f.write(b'{"traceEvents":[\n')
        
        first = True
        for event in event_list:
            event = _process_event_for_overlap(event, state)
            event = _update_flow_event_if_needed(event, state)
            
            if not first:
                f.write(b',\n')
            else:
                first = False
            f.write(orjson.dumps(event))
        
        for metadata_event in state.generate_overflow_metadata():
            f.write(b',\n')
            f.write(orjson.dumps(metadata_event))
        
        f.write(b'\n]}')


def _read_trace_events(input_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load traceEvents from a JSON or JSON.gz Chrome trace file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input trace not found: {input_path}")

    opener = gzip.open if input_path.suffix == '.gz' or input_path.name.endswith('.json.gz') else Path.open
    with opener(input_path, 'rb') as f:  # type: ignore[arg-type]
        trace_data = orjson.loads(f.read())

    extra_fields: dict[str, Any] = {}
    events: Any

    if isinstance(trace_data, dict):
        events = trace_data.get("traceEvents")
        if not isinstance(events, list):
            raise ValueError("Input trace must contain a 'traceEvents' list")
        extra_fields = {k: v for k, v in trace_data.items() if k != "traceEvents"}
    elif isinstance(trace_data, list):
        events = trace_data
    else:
        raise ValueError("Unsupported trace format. Expected dict with 'traceEvents' or a list of events.")

    return events, extra_fields


def _write_trace_with_metadata(output_path: Path, events: Iterable[dict], extra_fields: dict[str, Any]) -> None:
    """Write processed events back to disk while preserving non-traceEvents fields."""
    opener = gzip.open if output_path.suffix == '.gz' or output_path.name.endswith('.json.gz') else Path.open

    def _stream_to_file(f) -> None:
        f.write(b'{')

        extra_items = list(extra_fields.items())
        for idx, (key, value) in enumerate(extra_items):
            if idx > 0:
                f.write(b',')
            f.write(orjson.dumps(key))
            f.write(b':')
            f.write(orjson.dumps(value))

        if extra_items:
            f.write(b',')

        f.write(b'"traceEvents":[\n')

        first = True
        for event in events:
            if not first:
                f.write(b',\n')
            else:
                first = False
            f.write(orjson.dumps(event))

        f.write(b'\n]}')

    with opener(output_path, 'wb') as f:  # type: ignore[arg-type]
        _stream_to_file(f)


def process_chrome_trace_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Process a Chrome trace file to handle overlapping events.
    
    Reads a JSON or JSON.gz trace file, applies overlap detection to move
    partially overlapping events to virtual overflow tracks, and writes
    the processed trace.
    
    Args:
        input_path: Path to input trace file (.json or .json.gz)
        output_path: Path for output file. If None, overwrites the original file.

    Returns:
        Path to the processed trace file.
    """
    source_path = Path(input_path)

    # Create backup if it doesn't exist
    backup_path = Path(str(source_path) + '.bkup')
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)

    events, extra_fields = _read_trace_events(source_path)

    def _ts_value(event: dict) -> float:
        try:
            return float(event.get('ts', 0))
        except Exception:
            return 0.0

    events.sort(key=_ts_value)

    state = OverflowState()
    for event in events:
        state.extract_thread_metadata(event)

    def _processed_events() -> Iterator[dict]:
        for event in events:
            processed = _process_event_for_overlap(event, state)
            processed = _update_flow_event_if_needed(processed, state)
            yield processed
        for metadata_event in state.generate_overflow_metadata():
            yield metadata_event

    target_path = Path(output_path) if output_path is not None else source_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    _write_trace_with_metadata(target_path, _processed_events(), extra_fields)

    return str(target_path)
