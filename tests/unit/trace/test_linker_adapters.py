"""
Tests for ncompass.trace.converters.linker.adapters module.
"""

import unittest

from ncompass.trace.converters.linker.adapters import NsysTraceEventAdapter
from ncompass.trace.converters.models import ChromeTraceEvent


class TestNsysTraceEventAdapter(unittest.TestCase):
    """Test cases for NsysTraceEventAdapter class."""

    def test_get_time_range_valid(self):
        """Test get_time_range with valid X-phase event."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNotNone(result)
        self.assertEqual(result, (100000, 150000))

    def test_get_time_range_non_x_phase(self):
        """Test get_time_range with non-X phase event."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="event",
            ph="B",  # Begin phase
            cat="kernel",
            ts=100.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_get_time_range_missing_start_ns(self):
        """Test get_time_range with missing start_ns."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"end_ns": 150000}  # Missing start_ns
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_get_time_range_missing_end_ns(self):
        """Test get_time_range with missing end_ns."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000}  # Missing end_ns
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_get_time_range_none_start_ns(self):
        """Test get_time_range with None start_ns value."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": None, "end_ns": 150000}
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_get_time_range_none_end_ns(self):
        """Test get_time_range with None end_ns value."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000, "end_ns": None}
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_get_time_range_zero_duration(self):
        """Test get_time_range with zero duration."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=0.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000, "end_ns": 100000}
        )
        
        result = adapter.get_time_range(event)
        self.assertIsNotNone(result)
        self.assertEqual(result, (100000, 100000))

    def test_get_correlation_id_valid(self):
        """Test get_correlation_id with valid correlationId."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"correlationId": 12345, "start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_correlation_id(event)
        self.assertEqual(result, 12345)

    def test_get_correlation_id_missing(self):
        """Test get_correlation_id with missing correlationId."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_correlation_id(event)
        self.assertIsNone(result)

    def test_get_correlation_id_zero(self):
        """Test get_correlation_id with zero correlationId."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"correlationId": 0, "start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_correlation_id(event)
        self.assertEqual(result, 0)

    def test_get_event_id_basic(self):
        """Test get_event_id with complete event."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"deviceId": 0, "raw_tid": 5, "start_ns": 100000, "end_ns": 150000}
        )
        
        result = adapter.get_event_id(event)
        self.assertEqual(result, ("kernel", 100000, 0, 5))

    def test_get_event_id_missing_fields(self):
        """Test get_event_id with missing optional fields."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={}  # Missing all fields
        )
        
        result = adapter.get_event_id(event)
        self.assertEqual(result, ("kernel", None, None, None))

    def test_get_event_id_partial_fields(self):
        """Test get_event_id with some missing fields."""
        adapter = NsysTraceEventAdapter()
        event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=100.0,
            dur=50.0,
            pid="Device 0",
            tid="1",
            args={"deviceId": 0}  # Missing start_ns and raw_tid
        )
        
        result = adapter.get_event_id(event)
        self.assertEqual(result, ("kernel", None, 0, None))

