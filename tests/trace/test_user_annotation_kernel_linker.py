"""
Tests for ncompass.trace.converters.linker.user_annotation_linker module.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from ncompass.trace.converters.linker.user_annotation_linker import (
    link_user_annotation_to_kernels,
    _load_chrome_trace,
)
from ncompass.trace.converters.linker.adapters import ChromeTraceEventAdapter
from ncompass.trace.converters.linker.algorithms import find_overlapping_intervals


class TestExtractEventTimeRange(unittest.TestCase):
    """Test cases for ChromeTraceEventAdapter.get_time_range method."""

    def test_extract_event_time_range_valid(self):
        """Test ChromeTraceEventAdapter.get_time_range with valid X-phase event."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": 100.0,
            "dur": 50.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNotNone(result)
        self.assertEqual(result, (100.0, 150.0))

    def test_extract_event_time_range_non_x_phase(self):
        """Test ChromeTraceEventAdapter.get_time_range with non-X phase event."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "B",  # Begin phase
            "ts": 100.0,
            "dur": 50.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_extract_event_time_range_missing_ts(self):
        """Test ChromeTraceEventAdapter.get_time_range with missing ts field."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "dur": 50.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_extract_event_time_range_missing_dur(self):
        """Test ChromeTraceEventAdapter.get_time_range with missing dur field."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": 100.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_extract_event_time_range_none_ts(self):
        """Test ChromeTraceEventAdapter.get_time_range with None ts value."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": None,
            "dur": 50.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_extract_event_time_range_none_dur(self):
        """Test ChromeTraceEventAdapter.get_time_range with None dur value."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": 100.0,
            "dur": None,
        }
        result = adapter.get_time_range(event)
        self.assertIsNone(result)

    def test_extract_event_time_range_zero_duration(self):
        """Test ChromeTraceEventAdapter.get_time_range with zero duration."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": 100.0,
            "dur": 0.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNotNone(result)
        self.assertEqual(result, (100.0, 100.0))

    def test_extract_event_time_range_negative_duration(self):
        """Test ChromeTraceEventAdapter.get_time_range with negative duration."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "ph": "X",
            "ts": 100.0,
            "dur": -10.0,
        }
        result = adapter.get_time_range(event)
        self.assertIsNotNone(result)
        self.assertEqual(result, (100.0, 90.0))


class TestGetCorrelationId(unittest.TestCase):
    """Test cases for ChromeTraceEventAdapter.get_correlation_id method."""

    def test_get_correlation_id_with_correlation_field(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with 'correlation' field in args."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {
                "correlation": 12345,
            }
        }
        result = adapter.get_correlation_id(event)
        self.assertEqual(result, 12345)

    def test_get_correlation_id_with_correlation_id_field(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with 'correlationId' field in args."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {
                "correlationId": 67890,
            }
        }
        result = adapter.get_correlation_id(event)
        self.assertEqual(result, 67890)

    def test_get_correlation_id_both_fields_prefers_correlation(self):
        """Test ChromeTraceEventAdapter.get_correlation_id prefers 'correlation' over 'correlationId'."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {
                "correlation": 12345,
                "correlationId": 67890,
            }
        }
        result = adapter.get_correlation_id(event)
        self.assertEqual(result, 12345)

    def test_get_correlation_id_missing_args(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with missing args dict."""
        adapter = ChromeTraceEventAdapter()
        event = {}
        result = adapter.get_correlation_id(event)
        self.assertIsNone(result)

    def test_get_correlation_id_empty_args(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with empty args dict."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {}
        }
        result = adapter.get_correlation_id(event)
        self.assertIsNone(result)

    def test_get_correlation_id_missing_both_fields(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with neither correlation field."""
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {
                "other_field": "value",
            }
        }
        result = adapter.get_correlation_id(event)
        self.assertIsNone(result)

    def test_get_correlation_id_zero_correlation(self):
        """Test ChromeTraceEventAdapter.get_correlation_id with zero correlation ID.
        
        Note: Due to 'or' operator, 0 is falsy so it will fall back to correlationId.
        This tests the actual behavior of the code.
        """
        adapter = ChromeTraceEventAdapter()
        event = {
            "args": {
                "correlation": 0,
            }
        }
        result = adapter.get_correlation_id(event)
        # Due to 'or' operator, 0 is falsy so returns None (no correlationId)
        self.assertIsNone(result)
        
        # If correlationId is also 0, it will return 0
        event2 = {
            "args": {
                "correlation": 0,
                "correlationId": 0,
            }
        }
        result2 = adapter.get_correlation_id(event2)
        self.assertEqual(result2, 0)


class TestFindOverlappingCudaRuntime(unittest.TestCase):
    """Test cases for find_overlapping_intervals function."""

    def test_find_overlapping_cuda_runtime_basic_overlap(self):
        """Test basic overlapping intervals."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 1)
        event_id = ("test_annotation", 100.0, 1, 1)
        self.assertIn(event_id, result)
        self.assertEqual(len(result[event_id]), 1)

    def test_find_overlapping_cuda_runtime_no_overlap(self):
        """Test non-overlapping intervals."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 200.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 0)

    def test_find_overlapping_cuda_runtime_multiple_cuda_overlaps(self):
        """Test multiple CUDA runtime events overlapping with one user_annotation."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 100.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel1",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "cudaLaunchKernel2",
                "ph": "X",
                "ts": 150.0,
                "dur": 30.0,
                "pid": 1,
                "tid": 1,
            },
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 1)
        event_id = ("test_annotation", 100.0, 1, 1)
        self.assertIn(event_id, result)
        self.assertEqual(len(result[event_id]), 2)

    def test_find_overlapping_cuda_runtime_user_annotation_no_overlaps(self):
        """Test user_annotation events with no overlaps."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = []
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 0)

    def test_find_overlapping_cuda_runtime_invalid_time_range_filtered(self):
        """Test events with invalid time ranges are filtered out."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "invalid_annotation",
                "ph": "B",  # Not X phase, should be filtered
                "ts": 200.0,
                "pid": 1,
                "tid": 1,
            },
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "invalid_cuda",
                "ph": "X",
                "ts": 300.0,
                # Missing dur, should be filtered
                "pid": 1,
                "tid": 1,
            },
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        # Only valid user_annotation should be in result
        self.assertEqual(len(result), 1)
        event_id = ("test_annotation", 100.0, 1, 1)
        self.assertIn(event_id, result)

    def test_find_overlapping_cuda_runtime_touching_intervals(self):
        """Test touching intervals (end of one equals start of another)."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 150.0,  # Touches end of user_annotation
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        # Touching intervals don't overlap
        self.assertEqual(len(result), 0)

    def test_find_overlapping_cuda_runtime_nested_intervals(self):
        """Test nested intervals (CUDA runtime completely within user_annotation)."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "test_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 100.0,
                "pid": 1,
                "tid": 1,
            }
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 120.0,
                "dur": 30.0,  # Completely within user_annotation
                "pid": 1,
                "tid": 1,
            }
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 1)
        event_id = ("test_annotation", 100.0, 1, 1)
        self.assertIn(event_id, result)
        self.assertEqual(len(result[event_id]), 1)

    def test_find_overlapping_cuda_runtime_empty_inputs(self):
        """Test with empty input lists."""
        adapter = ChromeTraceEventAdapter()
        result = find_overlapping_intervals(
            [], [], adapter, "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 0)

    def test_find_overlapping_cuda_runtime_multiple_user_annotations(self):
        """Test multiple user_annotation events."""
        adapter = ChromeTraceEventAdapter()
        user_annotation_events = [
            {
                "name": "annotation1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "annotation2",
                "ph": "X",
                "ts": 200.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            },
        ]
        cuda_runtime_events = [
            {
                "name": "cudaLaunchKernel1",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "cudaLaunchKernel2",
                "ph": "X",
                "ts": 210.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
            },
        ]
        result = find_overlapping_intervals(
            user_annotation_events, cuda_runtime_events, adapter,
            "user_annotation", "cuda_runtime"
        )
        self.assertEqual(len(result), 2)
        self.assertIn(("annotation1", 100.0, 1, 1), result)
        self.assertIn(("annotation2", 200.0, 1, 1), result)


class TestLoadChromeTrace(unittest.TestCase):
    """Test cases for _load_chrome_trace function."""

    def test_load_chrome_trace_array_format(self):
        """Test loading array format trace."""
        trace_data = [
            {"name": "event1", "ph": "X", "ts": 100.0},
            {"name": "event2", "ph": "X", "ts": 200.0},
        ]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_path = f.name

        try:
            result = _load_chrome_trace(trace_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "event1")
            self.assertEqual(result[1]["name"], "event2")
        finally:
            Path(trace_path).unlink()

    def test_load_chrome_trace_object_format(self):
        """Test loading object format with traceEvents key."""
        trace_data = {
            "traceEvents": [
                {"name": "event1", "ph": "X", "ts": 100.0},
                {"name": "event2", "ph": "X", "ts": 200.0},
            ],
            "otherMetadata": {"key": "value"},
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_path = f.name

        try:
            result = _load_chrome_trace(trace_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["name"], "event1")
            self.assertEqual(result[1]["name"], "event2")
        finally:
            Path(trace_path).unlink()

    def test_load_chrome_trace_invalid_format(self):
        """Test invalid format raises ValueError."""
        trace_data = {"invalid": "format"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_path = f.name

        try:
            with self.assertRaises(ValueError) as context:
                _load_chrome_trace(trace_path)
            self.assertIn("Unexpected trace format", str(context.exception))
        finally:
            Path(trace_path).unlink()

    def test_load_chrome_trace_file_not_found(self):
        """Test file not found raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            _load_chrome_trace("/nonexistent/file.json")

    def test_load_chrome_trace_invalid_json(self):
        """Test invalid JSON raises JSONDecodeError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {")
            trace_path = f.name

        try:
            with self.assertRaises(json.JSONDecodeError):
                _load_chrome_trace(trace_path)
        finally:
            Path(trace_path).unlink()

    def test_load_chrome_trace_path_object(self):
        """Test loading with Path object instead of string."""
        trace_data = [{"name": "event1", "ph": "X", "ts": 100.0}]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_path = Path(f.name)

        try:
            result = _load_chrome_trace(trace_path)
            self.assertEqual(len(result), 1)
        finally:
            trace_path.unlink()


class TestLinkUserAnnotationToKernels(unittest.TestCase):
    """Test cases for link_user_annotation_to_kernels function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_files = []

    def tearDown(self):
        """Clean up test fixtures."""
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()

    def _create_trace_file(self, trace_data):
        """Helper to create a temporary trace file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(trace_data, f)
            trace_path = f.name
            self.temp_files.append(trace_path)
            return trace_path

    def test_link_user_annotation_to_kernels_successful_linking(self):
        """Test successful linking with complete chain."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should create a new gpu_user_annotation event and keep user_annotation
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        ua_events = [e for e in result if e.get("cat") == "user_annotation"]
        self.assertEqual(len(gpu_ua_events), 1)
        self.assertEqual(len(ua_events), 1)  # user_annotation should be kept
        self.assertEqual(gpu_ua_events[0]["name"], "forward")
        self.assertEqual(gpu_ua_events[0]["ts"], 120.0)
        self.assertEqual(gpu_ua_events[0]["dur"], 30.0)
        self.assertEqual(gpu_ua_events[0]["args"]["kernel_count"], 1)
        self.assertEqual(ua_events[0]["name"], "forward")

    def test_link_user_annotation_to_kernels_no_user_annotation_events(self):
        """Test with no user_annotation events (early return)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should return original trace unchanged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "kernel")

    def test_link_user_annotation_to_kernels_no_cuda_runtime_events(self):
        """Test with no cuda_runtime events (early return)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should return original trace unchanged
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "forward")

    def test_link_user_annotation_to_kernels_no_kernel_events(self):
        """Test with no kernel events (early return)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should return original trace unchanged
        self.assertEqual(len(result), 2)

    def test_link_user_annotation_to_kernels_replacement_both_exist(self):
        """Test replacement when both gpu_user_annotation and user_annotation exist."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "forward",
                    "cat": "gpu_user_annotation",
                    "ph": "X",
                    "ts": 90.0,
                    "dur": 60.0,
                    "pid": 0,
                    "tid": 0,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should replace old gpu_user_annotation with new one, but keep user_annotation
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        ua_events = [e for e in result if e.get("cat") == "user_annotation"]
        self.assertEqual(len(gpu_ua_events), 1)
        self.assertEqual(len(ua_events), 1)  # user_annotation should be kept
        self.assertEqual(ua_events[0]["name"], "forward")
        self.assertEqual(gpu_ua_events[0]["pid"], 0)  # Uses pid from existing gpu_ua

    def test_link_user_annotation_to_kernels_replacement_ua_only(self):
        """Test when only user_annotation exists - should keep it and add gpu_user_annotation."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should keep user_annotation and add gpu_user_annotation
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        ua_events = [e for e in result if e.get("cat") == "user_annotation"]
        self.assertEqual(len(gpu_ua_events), 1)
        self.assertEqual(len(ua_events), 1)  # user_annotation should be kept
        self.assertEqual(ua_events[0]["name"], "forward")

    def test_link_user_annotation_to_kernels_gpu_ua_only_no_replacement(self):
        """Test when only gpu_user_annotation exists (no replacement)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "gpu_user_annotation",
                    "ph": "X",
                    "ts": 90.0,
                    "dur": 60.0,
                    "pid": 0,
                    "tid": 0,
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should return original trace unchanged (no user_annotation to link)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "forward")

    def test_link_user_annotation_to_kernels_pid_tid_from_kernels_pytorch_format(self):
        """Test pid/tid assignment from kernels (PyTorch format)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 2},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 1)
        # Should use device from kernel args
        self.assertEqual(gpu_ua_events[0]["pid"], 2)
        self.assertEqual(gpu_ua_events[0]["tid"], 0)

    def test_link_user_annotation_to_kernels_pid_tid_from_kernels_nsys_format(self):
        """Test pid/tid assignment from kernels (nsys2chrome format)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": "Device 1",
                    "tid": 5,
                    "args": {"correlationId": 12345},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 1)
        # Should use Device X format from kernel
        self.assertEqual(gpu_ua_events[0]["pid"], "Device 1")
        self.assertEqual(gpu_ua_events[0]["tid"], 5)

    def test_link_user_annotation_to_kernels_multiple_kernels_same_correlation(self):
        """Test multiple kernels with same correlationId."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 200.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel1",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
                {
                    "name": "kernel2",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 150.0,
                    "dur": 40.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 1)
        # Should span from min start to max end
        self.assertEqual(gpu_ua_events[0]["ts"], 120.0)
        self.assertEqual(gpu_ua_events[0]["dur"], 70.0)  # 150 + 40 - 120
        self.assertEqual(gpu_ua_events[0]["args"]["kernel_count"], 2)

    def test_link_user_annotation_to_kernels_missing_correlation_ids(self):
        """Test events with missing correlation IDs."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    # Missing correlationId
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should not create linked event (no correlation)
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 0)

    def test_link_user_annotation_to_kernels_correlation_id_no_match(self):
        """Test correlation IDs that don't match kernels."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 99999},  # Doesn't match kernel
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},  # Different ID
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should not create linked event (no matching correlation)
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 0)

    def test_link_user_annotation_to_kernels_kernel_time_range_calculation(self):
        """Test kernel time range calculation (min start, max end)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 200.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel1",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 150.0,  # Later start
                    "dur": 20.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
                {
                    "name": "kernel2",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,  # Earlier start
                    "dur": 50.0,  # Later end
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 1)
        # Should use min start (120.0) and max end (120.0 + 50.0 = 170.0)
        self.assertEqual(gpu_ua_events[0]["ts"], 120.0)
        self.assertEqual(gpu_ua_events[0]["dur"], 50.0)

    @patch('ncompass.trace.converters.linker.user_annotation_linker.logger')
    def test_link_user_annotation_to_kernels_verbose_logging(self, mock_logger):
        """Test verbose logging output."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path, verbose=True)

        # Verify logger was called
        self.assertTrue(mock_logger.info.called)
        # Check that statistics were logged
        call_args_list = [str(call) for call in mock_logger.info.call_args_list]
        self.assertTrue(
            any("user_annotation events" in str(call) for call in call_args_list)
        )

    def test_link_user_annotation_to_kernels_no_new_events_created(self):
        """Test when no new events are created (no overlaps)."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 200.0,  # No overlap
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 210.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345, "device": 0},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        # Should return original trace unchanged
        self.assertEqual(len(result), 3)
        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 0)

    def test_link_user_annotation_to_kernels_fallback_pid_tid(self):
        """Test fallback pid/tid when not found in kernels."""
        trace_data = {
            "traceEvents": [
                {
                    "name": "forward",
                    "cat": "user_annotation",
                    "ph": "X",
                    "ts": 100.0,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                },
                {
                    "name": "cudaLaunchKernel",
                    "cat": "cuda_runtime",
                    "ph": "X",
                    "ts": 110.0,
                    "dur": 20.0,
                    "pid": 1,
                    "tid": 1,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel",
                    "cat": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    # Missing pid/tid and device
                    "args": {"correlationId": 12345},
                },
            ]
        }
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        self.assertEqual(len(gpu_ua_events), 1)
        # Should use fallback values
        self.assertEqual(gpu_ua_events[0]["pid"], 0)
        self.assertEqual(gpu_ua_events[0]["tid"], 0)

    def test_link_user_annotation_to_kernels_array_format_trace(self):
        """Test with array format trace (not object with traceEvents)."""
        trace_data = [
            {
                "name": "forward",
                "cat": "user_annotation",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 1,
                "tid": 1,
            },
            {
                "name": "cudaLaunchKernel",
                "cat": "cuda_runtime",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 12345},
            },
            {
                "name": "kernel",
                "cat": "kernel",
                "ph": "X",
                "ts": 120.0,
                "dur": 30.0,
                "pid": 0,
                "tid": 0,
                "args": {"correlationId": 12345, "device": 0},
            },
        ]
        trace_path = self._create_trace_file(trace_data)
        result = link_user_annotation_to_kernels(trace_path)

        gpu_ua_events = [
            e for e in result if e.get("cat") == "gpu_user_annotation"
        ]
        ua_events = [e for e in result if e.get("cat") == "user_annotation"]
        self.assertEqual(len(gpu_ua_events), 1)
        self.assertEqual(len(ua_events), 1)  # user_annotation should be kept

