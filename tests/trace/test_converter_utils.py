"""
Tests for ncompass.trace.converters.utils module.
"""

import unittest

from ncompass.trace.converters.utils import (
    ns_to_us,
    validate_chrome_trace,
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

