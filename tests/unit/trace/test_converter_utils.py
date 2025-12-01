"""
Tests for ncompass.trace.converters.utils module.
"""

import gzip
import json
import os
import tempfile
import unittest

from ncompass.trace.converters.utils import (
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
        events = {"traceEvents": []}
        
        write_chrome_trace_gz(output_path, events)
        
        # Verify file exists and is gzip-compressed
        self.assertTrue(os.path.exists(output_path))
        
        # Verify it's valid gzip by attempting to decompress
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            content = f.read()
        
        # Verify it's valid JSON
        parsed = json.loads(content)
        self.assertEqual(parsed, events)

    def test_write_chrome_trace_gz_readable(self):
        """Test that output can be read back and matches input."""
        output_path = os.path.join(self.temp_dir, "test.json.gz")
        events = {
            "traceEvents": [
                {"name": "event1", "ph": "X", "ts": 100.0, "dur": 50.0, "pid": 0, "tid": 0, "cat": "kernel"},
                {"name": "event2", "ph": "B", "ts": 200.0, "pid": 0, "tid": 0, "cat": "nvtx"},
            ]
        }
        
        write_chrome_trace_gz(output_path, events)
        
        # Read back and verify
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back, events)
        self.assertEqual(len(read_back["traceEvents"]), 2)
        self.assertEqual(read_back["traceEvents"][0]["name"], "event1")
        self.assertEqual(read_back["traceEvents"][1]["name"], "event2")

    def test_write_chrome_trace_gz_with_trace_events(self):
        """Test with realistic traceEvents structure including args and metadata."""
        output_path = os.path.join(self.temp_dir, "test.json.gz")
        events = {
            "traceEvents": [
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
        }
        
        write_chrome_trace_gz(output_path, events)
        
        # Verify file size is smaller than uncompressed would be (basic compression check)
        file_size = os.path.getsize(output_path)
        uncompressed_size = len(json.dumps(events).encode('utf-8'))
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
        events = {"traceEvents": []}
        
        write_chrome_trace_gz(output_path, events)
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back, {"traceEvents": []})

    def test_write_chrome_trace_gz_unicode_content(self):
        """Test that unicode content is properly encoded."""
        output_path = os.path.join(self.temp_dir, "unicode.json.gz")
        events = {
            "traceEvents": [
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
        }
        
        write_chrome_trace_gz(output_path, events)
        
        with gzip.open(output_path, 'rt', encoding='utf-8') as f:
            read_back = json.load(f)
        
        self.assertEqual(read_back["traceEvents"][0]["name"], "test_事件_émoji_🚀")
        self.assertEqual(read_back["traceEvents"][0]["args"]["description"], "テスト説明")

