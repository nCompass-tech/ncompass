"""
Tests for ncompass.trace.converters.utils module.
"""

import gzip
import json
import os
import tempfile
import unittest

from ncompass.trace.converters.utils import (
    _process_event_for_overlap,
    _OVERFLOW_PREFIX,
    ns_to_us,
    validate_chrome_trace,
    write_chrome_trace_gz,
)


class TestNsToUs(unittest.TestCase):
    """Test cases for ns_to_us function."""

    def test_ns_to_us_basic(self):
        """Test basic nanosecond to microsecond conversion."""
        result = ns_to_us(1000000)
        self.assertEqual(result, 1000.0)

    def test_ns_to_us_zero(self):
        """Test conversion of zero nanoseconds."""
        result = ns_to_us(0)
        self.assertEqual(result, 0.0)

    def test_ns_to_us_fractional(self):
        """Test conversion with fractional microseconds."""
        result = ns_to_us(1500)
        self.assertEqual(result, 1.5)

    def test_ns_to_us_large_value(self):
        """Test conversion of large nanosecond value."""
        result = ns_to_us(1000000000)
        self.assertEqual(result, 1000000.0)

    def test_ns_to_us_small_value(self):
        """Test conversion of small nanosecond value."""
        result = ns_to_us(1)
        self.assertEqual(result, 0.001)


class TestValidateChromeTrace(unittest.TestCase):
    """Test cases for validate_chrome_trace function."""

    def test_validate_chrome_trace_valid(self):
        """Test validation of valid Chrome trace events."""
        events = [
            {
                "name": "event1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
            {
                "name": "event2",
                "ph": "B",
                "ts": 200.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        result = validate_chrome_trace(events)
        self.assertTrue(result)

    def test_validate_chrome_trace_missing_required_field(self):
        """Test validation fails when required field is missing."""
        events = [
            {
                "name": "event1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                # Missing "tid"
                "cat": "kernel",
            },
        ]
        
        with self.assertRaises(ValueError) as cm:
            validate_chrome_trace(events)
        self.assertIn("missing required fields", str(cm.exception))
        self.assertIn("tid", str(cm.exception))

    def test_validate_chrome_trace_missing_name(self):
        """Test validation fails when name is missing."""
        events = [
            {
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        with self.assertRaises(ValueError) as cm:
            validate_chrome_trace(events)
        self.assertIn("missing required fields", str(cm.exception))
        self.assertIn("name", str(cm.exception))

    def test_validate_chrome_trace_invalid_phase(self):
        """Test validation fails with invalid phase."""
        events = [
            {
                "name": "event1",
                "ph": "Z",  # Invalid phase
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        with self.assertRaises(ValueError) as cm:
            validate_chrome_trace(events)
        self.assertIn("invalid phase", str(cm.exception))
        self.assertIn("Z", str(cm.exception))

    def test_validate_chrome_trace_missing_dur_for_x_phase(self):
        """Test validation fails when X phase event is missing dur."""
        events = [
            {
                "name": "event1",
                "ph": "X",
                "ts": 100.0,
                # Missing "dur" for X phase
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        with self.assertRaises(ValueError) as cm:
            validate_chrome_trace(events)
        self.assertIn("missing 'dur' field", str(cm.exception))

    def test_validate_chrome_trace_valid_phases(self):
        """Test validation accepts all valid phases."""
        valid_phases = ["B", "E", "X", "i", "C", "b", "n", "e", "s", "t", "f", "P", "N", "O", "D", "M", "V", "v", "R", "c", "(", ")"]
        
        for phase in valid_phases:
            events = [
                {
                    "name": "event1",
                    "ph": phase,
                    "ts": 100.0,
                    "pid": 0,
                    "tid": 0,
                    "cat": "kernel",
                },
            ]
            # Add dur for X phase
            if phase == "X":
                events[0]["dur"] = 50.0
            
            try:
                result = validate_chrome_trace(events)
                self.assertTrue(result)
            except ValueError as e:
                self.fail(f"Phase '{phase}' should be valid but raised: {e}")

    def test_validate_chrome_trace_empty_list(self):
        """Test validation with empty event list."""
        result = validate_chrome_trace([])
        self.assertTrue(result)

    def test_validate_chrome_trace_multiple_events(self):
        """Test validation with multiple events."""
        events = [
            {
                "name": "event1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
            {
                "name": "event2",
                "ph": "B",
                "ts": 200.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
            {
                "name": "event3",
                "ph": "E",
                "ts": 250.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        result = validate_chrome_trace(events)
        self.assertTrue(result)

    def test_validate_chrome_trace_error_message_includes_event_index(self):
        """Test that error messages include the event index."""
        events = [
            {
                "name": "event1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
            {
                "name": "event2",
                "ph": "X",
                "ts": 200.0,
                # Missing dur
                "pid": 0,
                "tid": 0,
                "cat": "kernel",
            },
        ]
        
        with self.assertRaises(ValueError) as cm:
            validate_chrome_trace(events)
        self.assertIn("Event 1", str(cm.exception))


class TestWriteChromeTraceGz(unittest.TestCase):
    """Test cases for write_chrome_trace_gz function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_chrome_trace_gz_basic(self):
        """Test that function writes valid gzip-compressed JSON."""
        output_path = os.path.join(self.temp_dir, "test.json.gz")
        events = []
        
        write_chrome_trace_gz(output_path, events)
        
        # Verify file exists and is gzip-compressed
        self.assertTrue(os.path.exists(output_path))
        
        # Verify it's valid gzip by attempting to decompress
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            content = f.read()
        
        # Verify it's valid JSON with traceEvents wrapper
        parsed = json.loads(content)
        self.assertEqual(parsed, {"traceEvents": []})

    def test_write_chrome_trace_gz_readable(self):
        """Test that output can be read back and matches input."""
        output_path = os.path.join(self.temp_dir, "test.json.gz")
        events = [
            {"name": "event1", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": 0, "tid": 0, "cat": "kernel"},
            {"name": "event2", "ph": "B", "ts": 200.0, "pid": 0, "tid": 0, "cat": "nvtx"},
        ]
        
        write_chrome_trace_gz(output_path, events)
        
        # Read back and verify
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back, {"traceEvents": events})
        self.assertEqual(len(read_back["traceEvents"]), 2)
        self.assertEqual(read_back["traceEvents"][0]["name"], "event1")
        self.assertEqual(read_back["traceEvents"][1]["name"], "event2")

    def test_write_chrome_trace_gz_with_trace_events(self):
        """Test with realistic traceEvents structure including args and metadata."""
        output_path = os.path.join(self.temp_dir, "test.json.gz")
        events = [
            {
                "name": "kernel_launch",
                "ph": "X",
                "ts": 1000.5,
                "dur": 250.75,
                "pid": "Device 0",
                "tid": "Stream 1",
                "cat": "cuda",
                "args": {
                    "deviceId": 0,
                    "streamId": 1,
                    "gridDim": [256, 1, 1],
                    "blockDim": [128, 1, 1],
                }
            },
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0.0,
                "pid": "Device 0",
                "tid": "",
                "cat": "__metadata",
                "args": {"name": "Device 0"}
            },
            {
                "name": "nvtx_range",
                "ph": "B",
                "ts": 500.0,
                "pid": "Device 0",
                "tid": "Thread 1",
                "cat": "nvtx",
                "args": {"color": "#FF0000"}
            },
        ]
        
        write_chrome_trace_gz(output_path, events)
        
        # Verify file size is smaller than uncompressed would be (basic compression check)
        file_size = os.path.getsize(output_path)
        uncompressed_size = len(json.dumps({"traceEvents": events}).encode('utf-8'))
        self.assertLess(file_size, uncompressed_size)
        
        # Verify content integrity
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(len(read_back["traceEvents"]), 3)
        
        # Check nested args are preserved
        kernel_event = read_back["traceEvents"][0]
        self.assertEqual(kernel_event["args"]["gridDim"], [256, 1, 1])
        self.assertEqual(kernel_event["args"]["blockDim"], [128, 1, 1])

    def test_write_chrome_trace_gz_empty_trace_events(self):
        """Test with empty traceEvents list."""
        output_path = os.path.join(self.temp_dir, "empty.json.gz")
        events = []
        
        write_chrome_trace_gz(output_path, events)
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back, {"traceEvents": []})

    def test_write_chrome_trace_gz_unicode_content(self):
        """Test that unicode content is properly encoded."""
        output_path = os.path.join(self.temp_dir, "unicode.json.gz")
        events = [
            {
                "name": "test_事件_émoji_🚀",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": "Device 0",
                "tid": "Thread 1",
                "cat": "test",
                "args": {"description": "テスト説明"}
            }
        ]
        
        write_chrome_trace_gz(output_path, events)
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back["traceEvents"][0]["name"], "test_事件_émoji_🚀")
        self.assertEqual(read_back["traceEvents"][0]["args"]["description"], "テスト説明")


class TestProcessEventForOverlap(unittest.TestCase):
    """Test cases for _process_event_for_overlap function (overlap detection)."""

    def test_no_overlap_sequential_events(self):
        """Test that sequential non-overlapping events keep original tid."""
        max_end = {}
        
        # Event A: ts=100, dur=50, ends at 150
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=160, dur=30, starts after A ends - no overlap
        event_b = {"name": "B", "ph": "X", "ts": 160.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")

    def test_no_overlap_exactly_adjacent(self):
        """Test that events exactly adjacent (no gap, no overlap) keep original tid."""
        max_end = {}
        
        # Event A: ts=100, dur=50, ends at 150
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=150, dur=30, starts exactly when A ends - no overlap
        event_b = {"name": "B", "ph": "X", "ts": 150.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")

    def test_fully_nested_event_keeps_original_tid(self):
        """Test that fully nested events keep original tid (Perfetto allows this)."""
        max_end = {}
        
        # Event A: ts=100, dur=100, ends at 200 (long event)
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=120, dur=30, ends at 150 - fully nested within A
        event_b = {"name": "B", "ph": "X", "ts": 120.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")  # Should NOT be moved

    def test_partial_overlap_moves_to_overflow(self):
        """Test that partial overlap moves event to overflow track."""
        max_end = {}
        
        # Event A: ts=100, dur=50, ends at 150
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=140, dur=30, ends at 170 - starts before A ends, ends after A ends
        event_b = {"name": "B", "ph": "X", "ts": 140.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], f"{_OVERFLOW_PREFIX}Stream 7")  # Should be moved

    def test_partial_overlap_small_overlap(self):
        """Test partial overlap with very small overlap (like real GPU traces)."""
        max_end = {}
        
        # Event A: ts=9659065, dur=976, ends at 9660041
        event_a = {"name": "device_kernel", "ph": "X", "ts": 9659065.0, "dur": 976.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=9660039, dur=33, ends at 9660072 - overlaps by ~2µs
        event_b = {"name": "device_kernel", "ph": "X", "ts": 9660039.0, "dur": 33.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], f"{_OVERFLOW_PREFIX}Stream 7")

    def test_different_tracks_independent(self):
        """Test that events on different tracks don't affect each other."""
        max_end = {}
        
        # Event on Stream 7
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event on Stream 8 at overlapping time - should NOT be affected
        event_b = {"name": "B", "ph": "X", "ts": 120.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 8", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 8")  # Different track, unaffected

    def test_different_pids_independent(self):
        """Test that events on different pids don't affect each other."""
        max_end = {}
        
        # Event on Device 0
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event on Device 1 at overlapping time - should NOT be affected
        event_b = {"name": "B", "ph": "X", "ts": 120.0, "dur": 50.0, "pid": "Device 1", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")  # Different pid, unaffected

    def test_non_x_phase_events_unchanged(self):
        """Test that non-X phase events are not processed."""
        max_end = {}
        
        # Setup: add an event to establish max_end
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        _process_event_for_overlap(event_a, max_end)
        
        # B phase event at overlapping time - should NOT be moved
        event_b = {"name": "B", "ph": "B", "ts": 120.0, "pid": "Device 0", "tid": "Stream 7", "cat": "nvtx"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")  # Not an X event, unchanged

    def test_event_without_dur_unchanged(self):
        """Test that events without dur field are not processed."""
        max_end = {}
        
        # Setup: add an event to establish max_end
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        _process_event_for_overlap(event_a, max_end)
        
        # X event without dur - should NOT be moved (invalid, but shouldn't crash)
        event_b = {"name": "B", "ph": "X", "ts": 120.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], "Stream 7")

    def test_overflow_track_reused_after_gap(self):
        """Test that original track is reused after gap (no infinite tracks)."""
        max_end = {}
        
        # Event A: ts=100, dur=50, ends at 150
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_a = _process_event_for_overlap(event_a, max_end)
        self.assertEqual(result_a["tid"], "Stream 7")
        
        # Event B: ts=140, dur=30, ends at 170 - partial overlap, moves to overflow
        event_b = {"name": "B", "ph": "X", "ts": 140.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        self.assertEqual(result_b["tid"], f"{_OVERFLOW_PREFIX}Stream 7")
        
        # Event C: ts=200, dur=20 - starts after both A and B end, should go back to original
        event_c = {"name": "C", "ph": "X", "ts": 200.0, "dur": 20.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        result_c = _process_event_for_overlap(event_c, max_end)
        self.assertEqual(result_c["tid"], "Stream 7")  # Back to original track

    def test_does_not_mutate_original_event(self):
        """Test that original event is not mutated when moved to overflow."""
        max_end = {}
        
        # Event A
        event_a = {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"}
        _process_event_for_overlap(event_a, max_end)
        
        # Event B that will be moved - keep a reference to original tid
        original_tid = "Stream 7"
        event_b = {"name": "B", "ph": "X", "ts": 140.0, "dur": 30.0, "pid": "Device 0", "tid": original_tid, "cat": "kernel"}
        result_b = _process_event_for_overlap(event_b, max_end)
        
        # Result should have new tid
        self.assertEqual(result_b["tid"], f"{_OVERFLOW_PREFIX}Stream 7")
        # Original event should be unchanged (it's a copy)
        self.assertEqual(event_b["tid"], original_tid)


class TestWriteChromeTraceGzOverlapHandling(unittest.TestCase):
    """Test cases for overlap handling in write_chrome_trace_gz."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_handles_partial_overlap(self):
        """Test that write_chrome_trace_gz moves overlapping events to overflow track."""
        output_path = os.path.join(self.temp_dir, "overlap.json.gz")
        
        events = [
            {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},
            {"name": "B", "ph": "X", "ts": 140.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},  # Overlaps
        ]
        
        write_chrome_trace_gz(output_path, iter(events))
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        # First event should be on original track
        self.assertEqual(read_back["traceEvents"][0]["tid"], "Stream 7")
        # Second event should be on overflow track
        self.assertEqual(read_back["traceEvents"][1]["tid"], f"{_OVERFLOW_PREFIX}Stream 7")

    def test_write_preserves_non_overlapping_events(self):
        """Test that non-overlapping events keep original tid."""
        output_path = os.path.join(self.temp_dir, "no_overlap.json.gz")
        
        events = [
            {"name": "A", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},
            {"name": "B", "ph": "X", "ts": 200.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},  # No overlap
        ]
        
        write_chrome_trace_gz(output_path, iter(events))
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        # Both events should be on original track
        self.assertEqual(read_back["traceEvents"][0]["tid"], "Stream 7")
        self.assertEqual(read_back["traceEvents"][1]["tid"], "Stream 7")

    def test_write_preserves_fully_nested_events(self):
        """Test that fully nested events keep original tid."""
        output_path = os.path.join(self.temp_dir, "nested.json.gz")
        
        events = [
            {"name": "A", "ph": "X", "ts": 100.0, "dur": 100.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},
            {"name": "B", "ph": "X", "ts": 120.0, "dur": 30.0, "pid": "Device 0", "tid": "Stream 7", "cat": "kernel"},  # Nested
        ]
        
        write_chrome_trace_gz(output_path, iter(events))
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        # Both events should be on original track (nested is OK)
        self.assertEqual(read_back["traceEvents"][0]["tid"], "Stream 7")
        self.assertEqual(read_back["traceEvents"][1]["tid"], "Stream 7")

