"""
Tests for ncompass.trace.converters.linker.algorithms module.
"""

import unittest

from ncompass.trace.converters.linker.algorithms import (
    find_overlapping_intervals,
    build_correlation_map,
    aggregate_kernel_times,
    find_kernels_for_annotation,
)
from ncompass.trace.converters.linker.adapters import (
    ChromeTraceEventAdapter,
    NsysTraceEventAdapter,
)
from ncompass.trace.converters.models import ChromeTraceEvent


class TestBuildCorrelationMap(unittest.TestCase):
    """Test cases for build_correlation_map function."""

    def test_build_correlation_map_basic(self):
        """Test basic correlation map building."""
        adapter = ChromeTraceEventAdapter()
        kernel_events = [
            {
                "name": "kernel1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "args": {"correlationId": 12345},
            },
            {
                "name": "kernel2",
                "ph": "X",
                "ts": 150.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "args": {"correlationId": 12345},  # Same correlation ID
            },
            {
                "name": "kernel3",
                "ph": "X",
                "ts": 200.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "args": {"correlationId": 67890},  # Different correlation ID
            },
        ]
        
        correlation_map = build_correlation_map(kernel_events, adapter)
        
        self.assertIn(12345, correlation_map)
        self.assertIn(67890, correlation_map)
        self.assertEqual(len(correlation_map[12345]), 2)
        self.assertEqual(len(correlation_map[67890]), 1)

    def test_build_correlation_map_missing_correlation_id(self):
        """Test correlation map with events missing correlation IDs."""
        adapter = ChromeTraceEventAdapter()
        kernel_events = [
            {
                "name": "kernel1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "args": {"correlationId": 12345},
            },
            {
                "name": "kernel2",
                "ph": "X",
                "ts": 150.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
                "args": {},  # Missing correlation ID
            },
        ]
        
        correlation_map = build_correlation_map(kernel_events, adapter)
        
        self.assertIn(12345, correlation_map)
        self.assertEqual(len(correlation_map[12345]), 1)
        # Kernel without correlation ID should not be in map

    def test_build_correlation_map_empty_list(self):
        """Test correlation map with empty kernel list."""
        adapter = ChromeTraceEventAdapter()
        correlation_map = build_correlation_map([], adapter)
        
        self.assertEqual(len(correlation_map), 0)

    def test_build_correlation_map_nsys_adapter(self):
        """Test correlation map building with NsysTraceEventAdapter."""
        adapter = NsysTraceEventAdapter()
        kernel_events = [
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=100.0,
                dur=50.0,
                pid="Device 0",
                tid="1",
                args={"correlationId": 12345, "start_ns": 100000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="kernel2",
                ph="X",
                cat="kernel",
                ts=150.0,
                dur=50.0,
                pid="Device 0",
                tid="1",
                args={"correlationId": 12345, "start_ns": 150000, "end_ns": 200000}
            ),
        ]
        
        correlation_map = build_correlation_map(kernel_events, adapter)
        
        self.assertIn(12345, correlation_map)
        self.assertEqual(len(correlation_map[12345]), 2)


class TestAggregateKernelTimes(unittest.TestCase):
    """Test cases for aggregate_kernel_times function."""

    def test_aggregate_kernel_times_basic(self):
        """Test basic kernel time aggregation."""
        adapter = ChromeTraceEventAdapter()
        kernels = [
            {
                "name": "kernel1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
            },
            {
                "name": "kernel2",
                "ph": "X",
                "ts": 120.0,  # Earlier start
                "dur": 80.0,  # Later end
                "pid": 0,
                "tid": 0,
            },
        ]
        
        time_range = aggregate_kernel_times(kernels, adapter)
        
        self.assertIsNotNone(time_range)
        start, end = time_range
        self.assertEqual(start, 100.0)  # Min start
        self.assertEqual(end, 200.0)  # Max end (120 + 80)

    def test_aggregate_kernel_times_single_kernel(self):
        """Test aggregation with single kernel."""
        adapter = ChromeTraceEventAdapter()
        kernels = [
            {
                "name": "kernel1",
                "ph": "X",
                "ts": 100.0,
                "dur": 50.0,
                "pid": 0,
                "tid": 0,
            },
        ]
        
        time_range = aggregate_kernel_times(kernels, adapter)
        
        self.assertIsNotNone(time_range)
        start, end = time_range
        self.assertEqual(start, 100.0)
        self.assertEqual(end, 150.0)

    def test_aggregate_kernel_times_empty_list(self):
        """Test aggregation with empty kernel list."""
        adapter = ChromeTraceEventAdapter()
        time_range = aggregate_kernel_times([], adapter)
        
        self.assertIsNone(time_range)

    def test_aggregate_kernel_times_invalid_events(self):
        """Test aggregation with invalid events (non-X phase)."""
        adapter = ChromeTraceEventAdapter()
        kernels = [
            {
                "name": "kernel1",
                "ph": "B",  # Not X phase
                "ts": 100.0,
                "pid": 0,
                "tid": 0,
            },
        ]
        
        time_range = aggregate_kernel_times(kernels, adapter)
        
        self.assertIsNone(time_range)

    def test_aggregate_kernel_times_nsys_adapter(self):
        """Test aggregation with NsysTraceEventAdapter."""
        adapter = NsysTraceEventAdapter()
        kernels = [
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=100.0,
                dur=50.0,
                pid="Device 0",
                tid="1",
                args={"start_ns": 100000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="kernel2",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=80.0,
                pid="Device 0",
                tid="1",
                args={"start_ns": 120000, "end_ns": 200000}
            ),
        ]
        
        time_range = aggregate_kernel_times(kernels, adapter)
        
        self.assertIsNotNone(time_range)
        start, end = time_range
        self.assertEqual(start, 100000)  # Min start_ns
        self.assertEqual(end, 200000)  # Max end_ns


class TestFindKernelsForAnnotation(unittest.TestCase):
    """Test cases for find_kernels_for_annotation function."""

    def test_find_kernels_for_annotation_basic(self):
        """Test basic kernel finding."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 12345},
            },
        ]
        correlation_map = {
            12345: [
                {
                    "name": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 1)
        self.assertEqual(found_kernels[0]["name"], "kernel")

    def test_find_kernels_for_annotation_multiple_kernels(self):
        """Test finding multiple kernels from same correlation ID."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 12345},
            },
        ]
        correlation_map = {
            12345: [
                {
                    "name": "kernel1",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
                {
                    "name": "kernel2",
                    "ph": "X",
                    "ts": 150.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 2)

    def test_find_kernels_for_annotation_multiple_api_events(self):
        """Test finding kernels from multiple API events."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel1",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 12345},
            },
            {
                "name": "cudaLaunchKernel2",
                "ph": "X",
                "ts": 150.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 67890},
            },
        ]
        correlation_map = {
            12345: [
                {
                    "name": "kernel1",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ],
            67890: [
                {
                    "name": "kernel2",
                    "ph": "X",
                    "ts": 160.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 67890},
                },
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 2)

    def test_find_kernels_for_annotation_no_match(self):
        """Test when API events don't match any kernels."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 99999},  # Not in correlation_map
            },
        ]
        correlation_map = {
            12345: [
                {
                    "name": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 0)

    def test_find_kernels_for_annotation_missing_correlation_id(self):
        """Test when API events are missing correlation IDs."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {},  # Missing correlationId
            },
        ]
        correlation_map = {
            12345: [
                {
                    "name": "kernel",
                    "ph": "X",
                    "ts": 120.0,
                    "dur": 30.0,
                    "pid": 0,
                    "tid": 0,
                    "args": {"correlationId": 12345},
                },
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 0)

    def test_find_kernels_for_annotation_empty_kernels_list(self):
        """Test when correlation map has empty kernel lists."""
        adapter = ChromeTraceEventAdapter()
        overlapping_api_events = [
            {
                "name": "cudaLaunchKernel",
                "ph": "X",
                "ts": 110.0,
                "dur": 20.0,
                "pid": 1,
                "tid": 1,
                "args": {"correlationId": 12345},
            },
        ]
        correlation_map = {
            12345: [],  # Empty kernel list
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        # Should skip API calls that didn't launch kernels
        self.assertEqual(len(found_kernels), 0)

    def test_find_kernels_for_annotation_nsys_adapter(self):
        """Test kernel finding with NsysTraceEventAdapter."""
        adapter = NsysTraceEventAdapter()
        overlapping_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 110000, "end_ns": 120000}
            ),
        ]
        correlation_map = {
            12345: [
                ChromeTraceEvent(
                    name="kernel",
                    ph="X",
                    cat="kernel",
                    ts=120.0,
                    dur=30.0,
                    pid="Device 0",
                    tid="2",
                    args={"deviceId": 0, "correlationId": 12345, "start_ns": 120000, "end_ns": 150000}
                ),
            ],
        }
        
        found_kernels = find_kernels_for_annotation(
            overlapping_api_events, correlation_map, adapter
        )
        
        self.assertEqual(len(found_kernels), 1)
        self.assertEqual(found_kernels[0].name, "kernel")


class TestFindOverlappingIntervals(unittest.TestCase):
    """Test cases for find_overlapping_intervals function with NsysTraceEventAdapter."""

    def test_find_overlapping_intervals_nsys_adapter_basic(self):
        """Test overlapping intervals with NsysTraceEventAdapter."""
        adapter = NsysTraceEventAdapter()
        source_events = [
            ChromeTraceEvent(
                name="nvtx1",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000}
            ),
        ]
        target_events = [
            ChromeTraceEvent(
                name="cuda_api1",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=20.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "start_ns": 110000, "end_ns": 130000}
            ),
        ]
        
        overlap_map = find_overlapping_intervals(
            source_events, target_events, adapter, "nvtx", "cuda_api"
        )
        
        self.assertEqual(len(overlap_map), 1)
        event_id = ("nvtx1", 100000, 0, None)
        self.assertIn(event_id, overlap_map)
        self.assertEqual(len(overlap_map[event_id]), 1)

    def test_find_overlapping_intervals_nsys_adapter_no_overlap(self):
        """Test non-overlapping intervals with NsysTraceEventAdapter."""
        adapter = NsysTraceEventAdapter()
        source_events = [
            ChromeTraceEvent(
                name="nvtx1",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000}
            ),
        ]
        target_events = [
            ChromeTraceEvent(
                name="cuda_api1",
                ph="X",
                cat="cuda_api",
                ts=200.0,
                dur=20.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "start_ns": 200000, "end_ns": 220000}
            ),
        ]
        
        overlap_map = find_overlapping_intervals(
            source_events, target_events, adapter, "nvtx", "cuda_api"
        )
        
        self.assertEqual(len(overlap_map), 0)

    def test_find_overlapping_intervals_nsys_adapter_invalid_time_range(self):
        """Test that events with invalid time ranges are filtered."""
        adapter = NsysTraceEventAdapter()
        source_events = [
            ChromeTraceEvent(
                name="nvtx_valid",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="nvtx_invalid",
                ph="B",  # Not X phase
                cat="nvtx",
                ts=200.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0}
            ),
        ]
        target_events = [
            ChromeTraceEvent(
                name="cuda_api1",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=20.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "start_ns": 110000, "end_ns": 130000}
            ),
        ]
        
        overlap_map = find_overlapping_intervals(
            source_events, target_events, adapter, "nvtx", "cuda_api"
        )
        
        # Only valid source event should be in result
        self.assertEqual(len(overlap_map), 1)

