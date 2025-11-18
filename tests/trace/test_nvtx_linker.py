"""
Tests for ncompass.trace.converters.linker.nvtx_linker module.
"""

import unittest
from unittest.mock import patch

from ncompass.trace.converters.linker.nvtx_linker import (
    link_nvtx_to_kernels,
    _create_flow_events,
    _group_events_by_device,
    _build_correlation_map_with_cuda_api,
    _generate_flow_events_for_correlation_map,
    _create_nvtx_kernel_event,
    _process_device_nvtx_events,
)
from ncompass.trace.converters.models import ChromeTraceEvent, ConversionOptions
from ncompass.trace.converters.linker.adapters import NsysTraceEventAdapter


class TestCreateFlowEvents(unittest.TestCase):
    """Test cases for _create_flow_events function."""

    def test_create_flow_events_basic(self):
        """Test basic flow event creation."""
        cuda_api_event = ChromeTraceEvent(
            name="cudaLaunchKernel",
            ph="X",
            cat="cuda_api",
            ts=100.0,
            dur=10.0,
            pid="Device 0",
            tid="1",
            args={"correlationId": 12345}
        )
        kernel_event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=120.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"correlationId": 12345}
        )
        
        flow_start, flow_finish = _create_flow_events(
            cuda_api_event, kernel_event, 12345
        )
        
        self.assertEqual(flow_start.ph, "s")
        self.assertEqual(flow_start.cat, "cuda_flow")
        self.assertEqual(flow_start.ts, 100.0)
        self.assertEqual(flow_start.id, 12345)
        self.assertEqual(flow_start.pid, "Device 0")
        self.assertEqual(flow_start.tid, "1")
        
        self.assertEqual(flow_finish.ph, "f")
        self.assertEqual(flow_finish.cat, "cuda_flow")
        self.assertEqual(flow_finish.ts, 120.0)
        self.assertEqual(flow_finish.id, 12345)
        self.assertEqual(flow_finish.pid, "Device 0")
        self.assertEqual(flow_finish.tid, "2")
        self.assertEqual(flow_finish.bp, "e")

    def test_create_flow_events_different_correlation_id(self):
        """Test flow events with different correlation ID parameter."""
        cuda_api_event = ChromeTraceEvent(
            name="cudaLaunchKernel",
            ph="X",
            cat="cuda_api",
            ts=100.0,
            dur=10.0,
            pid="Device 0",
            tid="1",
            args={"correlationId": 12345}
        )
        kernel_event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=120.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"correlationId": 12345}
        )
        
        # Use different correlation ID in function call
        flow_start, flow_finish = _create_flow_events(
            cuda_api_event, kernel_event, 99999
        )
        
        self.assertEqual(flow_start.id, 99999)
        self.assertEqual(flow_finish.id, 99999)


class TestGroupEventsByDevice(unittest.TestCase):
    """Test cases for _group_events_by_device function."""

    def test_group_events_by_device_basic(self):
        """Test basic grouping by device ID."""
        nvtx_events = [
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
            ChromeTraceEvent(
                name="nvtx2",
                ph="X",
                cat="nvtx",
                ts=200.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 1, "start_ns": 200000, "end_ns": 250000}
            ),
        ]
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345}
            ),
        ]
        
        per_device_nvtx, per_device_cuda_api, per_device_kernels = _group_events_by_device(
            nvtx_events, cuda_api_events, kernel_events
        )
        
        self.assertEqual(len(per_device_nvtx[0]), 1)
        self.assertEqual(len(per_device_nvtx[1]), 1)
        self.assertEqual(len(per_device_cuda_api[0]), 1)
        self.assertEqual(len(per_device_kernels[0]), 1)

    def test_group_events_by_device_filters_incomplete_nvtx(self):
        """Test that incomplete NVTX events are filtered out."""
        nvtx_events = [
            ChromeTraceEvent(
                name="nvtx_complete",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="nvtx_missing_start",
                ph="X",
                cat="nvtx",
                ts=200.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "end_ns": 250000}  # Missing start_ns
            ),
            ChromeTraceEvent(
                name="nvtx_missing_device",
                ph="X",
                cat="nvtx",
                ts=300.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"start_ns": 300000, "end_ns": 350000}  # Missing deviceId
            ),
        ]
        
        per_device_nvtx, _, _ = _group_events_by_device(
            nvtx_events, [], []
        )
        
        self.assertEqual(len(per_device_nvtx[0]), 1)
        self.assertEqual(per_device_nvtx[0][0].name, "nvtx_complete")

    def test_group_events_by_device_filters_incomplete_cuda_api(self):
        """Test that CUDA API events without device ID or correlation ID are filtered."""
        cuda_api_events = [
            ChromeTraceEvent(
                name="cuda_complete",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345}
            ),
            ChromeTraceEvent(
                name="cuda_missing_device",
                ph="X",
                cat="cuda_api",
                ts=120.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"correlationId": 12346}  # Missing deviceId
            ),
            ChromeTraceEvent(
                name="cuda_missing_correlation",
                ph="X",
                cat="cuda_api",
                ts=130.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0}  # Missing correlationId
            ),
        ]
        
        _, per_device_cuda_api, _ = _group_events_by_device(
            [], cuda_api_events, []
        )
        
        self.assertEqual(len(per_device_cuda_api[0]), 1)
        self.assertEqual(per_device_cuda_api[0][0].name, "cuda_complete")

    def test_group_events_by_device_filters_incomplete_kernels(self):
        """Test that kernel events without device ID or correlation ID are filtered."""
        kernel_events = [
            ChromeTraceEvent(
                name="kernel_complete",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345}
            ),
            ChromeTraceEvent(
                name="kernel_missing_device",
                ph="X",
                cat="kernel",
                ts=130.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"correlationId": 12346}  # Missing deviceId
            ),
        ]
        
        _, _, per_device_kernels = _group_events_by_device(
            [], [], kernel_events
        )
        
        self.assertEqual(len(per_device_kernels[0]), 1)
        self.assertEqual(per_device_kernels[0][0].name, "kernel_complete")


class TestBuildCorrelationMapWithCudaApi(unittest.TestCase):
    """Test cases for _build_correlation_map_with_cuda_api function."""

    def test_build_correlation_map_with_cuda_api_basic(self):
        """Test basic correlation map building."""
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345}
            ),
            ChromeTraceEvent(
                name="kernel2",
                ph="X",
                cat="kernel",
                ts=150.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345}
            ),
        ]
        
        adapter = NsysTraceEventAdapter()
        correlation_map = _build_correlation_map_with_cuda_api(
            cuda_api_events, kernel_events, adapter
        )
        
        self.assertIn(12345, correlation_map)
        self.assertIsNotNone(correlation_map[12345]["cuda_api"])
        self.assertEqual(len(correlation_map[12345]["kernels"]), 2)
        self.assertEqual(correlation_map[12345]["cuda_api"].name, "cudaLaunchKernel")

    def test_build_correlation_map_with_cuda_api_multiple_correlations(self):
        """Test correlation map with multiple correlation IDs."""
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel1",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345}
            ),
            ChromeTraceEvent(
                name="cudaLaunchKernel2",
                ph="X",
                cat="cuda_api",
                ts=200.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 67890}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345}
            ),
            ChromeTraceEvent(
                name="kernel2",
                ph="X",
                cat="kernel",
                ts=210.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 67890}
            ),
        ]
        
        adapter = NsysTraceEventAdapter()
        correlation_map = _build_correlation_map_with_cuda_api(
            cuda_api_events, kernel_events, adapter
        )
        
        self.assertIn(12345, correlation_map)
        self.assertIn(67890, correlation_map)
        self.assertEqual(len(correlation_map[12345]["kernels"]), 1)
        self.assertEqual(len(correlation_map[67890]["kernels"]), 1)

    def test_build_correlation_map_with_cuda_api_no_kernels(self):
        """Test correlation map when CUDA API has no associated kernels."""
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345}
            ),
        ]
        kernel_events = []  # No kernels
        
        adapter = NsysTraceEventAdapter()
        correlation_map = _build_correlation_map_with_cuda_api(
            cuda_api_events, kernel_events, adapter
        )
        
        self.assertIn(12345, correlation_map)
        self.assertIsNotNone(correlation_map[12345]["cuda_api"])
        self.assertEqual(len(correlation_map[12345]["kernels"]), 0)


class TestGenerateFlowEventsForCorrelationMap(unittest.TestCase):
    """Test cases for _generate_flow_events_for_correlation_map function."""

    def test_generate_flow_events_basic(self):
        """Test basic flow event generation."""
        cuda_api_event = ChromeTraceEvent(
            name="cudaLaunchKernel",
            ph="X",
            cat="cuda_api",
            ts=110.0,
            dur=10.0,
            pid="Device 0",
            tid="1",
            args={"deviceId": 0, "correlationId": 12345}
        )
        kernel_event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=120.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"deviceId": 0, "correlationId": 12345}
        )
        
        correlation_map = {
            12345: {
                "cuda_api": cuda_api_event,
                "kernels": [kernel_event]
            }
        }
        
        flow_events = _generate_flow_events_for_correlation_map(correlation_map)
        
        self.assertEqual(len(flow_events), 2)  # Start and finish
        self.assertEqual(flow_events[0].ph, "s")
        self.assertEqual(flow_events[1].ph, "f")

    def test_generate_flow_events_multiple_kernels(self):
        """Test flow event generation with multiple kernels."""
        cuda_api_event = ChromeTraceEvent(
            name="cudaLaunchKernel",
            ph="X",
            cat="cuda_api",
            ts=110.0,
            dur=10.0,
            pid="Device 0",
            tid="1",
            args={"deviceId": 0, "correlationId": 12345}
        )
        kernel1 = ChromeTraceEvent(
            name="kernel1",
            ph="X",
            cat="kernel",
            ts=120.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"deviceId": 0, "correlationId": 12345}
        )
        kernel2 = ChromeTraceEvent(
            name="kernel2",
            ph="X",
            cat="kernel",
            ts=150.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"deviceId": 0, "correlationId": 12345}
        )
        
        correlation_map = {
            12345: {
                "cuda_api": cuda_api_event,
                "kernels": [kernel1, kernel2]
            }
        }
        
        flow_events = _generate_flow_events_for_correlation_map(correlation_map)
        
        # Should have 2 flow pairs (4 events total)
        self.assertEqual(len(flow_events), 4)
        self.assertEqual(flow_events[0].ph, "s")
        self.assertEqual(flow_events[1].ph, "f")
        self.assertEqual(flow_events[2].ph, "s")
        self.assertEqual(flow_events[3].ph, "f")

    def test_generate_flow_events_no_cuda_api(self):
        """Test flow event generation when CUDA API is missing."""
        kernel_event = ChromeTraceEvent(
            name="kernel",
            ph="X",
            cat="kernel",
            ts=120.0,
            dur=30.0,
            pid="Device 0",
            tid="2",
            args={"deviceId": 0, "correlationId": 12345}
        )
        
        correlation_map = {
            12345: {
                "cuda_api": None,
                "kernels": [kernel_event]
            }
        }
        
        flow_events = _generate_flow_events_for_correlation_map(correlation_map)
        
        # Should not generate flow events without CUDA API
        self.assertEqual(len(flow_events), 0)

    def test_generate_flow_events_no_kernels(self):
        """Test flow event generation when no kernels exist."""
        cuda_api_event = ChromeTraceEvent(
            name="cudaLaunchKernel",
            ph="X",
            cat="cuda_api",
            ts=110.0,
            dur=10.0,
            pid="Device 0",
            tid="1",
            args={"deviceId": 0, "correlationId": 12345}
        )
        
        correlation_map = {
            12345: {
                "cuda_api": cuda_api_event,
                "kernels": []
            }
        }
        
        flow_events = _generate_flow_events_for_correlation_map(correlation_map)
        
        # Should not generate flow events without kernels
        self.assertEqual(len(flow_events), 0)


class TestCreateNvtxKernelEvent(unittest.TestCase):
    """Test cases for _create_nvtx_kernel_event function."""

    def test_create_nvtx_kernel_event_basic(self):
        """Test basic nvtx-kernel event creation."""
        nvtx_event = ChromeTraceEvent(
            name="forward",
            ph="X",
            cat="nvtx",
            ts=100.0,
            dur=50.0,
            pid="CPU",
            tid="1",
            args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
        )
        options = ConversionOptions()
        
        event = _create_nvtx_kernel_event(
            nvtx_event, 120000, 150000, 0, options
        )
        
        self.assertEqual(event.name, "forward")
        self.assertEqual(event.ph, "X")
        self.assertEqual(event.cat, "nvtx-kernel")
        self.assertEqual(event.ts, 120.0)  # ns_to_us(120000)
        self.assertEqual(event.dur, 30.0)  # ns_to_us(150000 - 120000)
        self.assertEqual(event.pid, "Device 0")
        self.assertEqual(event.tid, "NVTX Kernel Thread 1")

    def test_create_nvtx_kernel_event_with_color_scheme(self):
        """Test nvtx-kernel event creation with color scheme."""
        nvtx_event = ChromeTraceEvent(
            name="compute_forward",
            ph="X",
            cat="nvtx",
            ts=100.0,
            dur=50.0,
            pid="CPU",
            tid="1",
            args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
        )
        options = ConversionOptions(
            nvtx_color_scheme={"compute": "thread_state_running"}
        )
        
        event = _create_nvtx_kernel_event(
            nvtx_event, 120000, 150000, 0, options
        )
        
        self.assertEqual(event.cname, "thread_state_running")

    def test_create_nvtx_kernel_event_color_scheme_no_match(self):
        """Test nvtx-kernel event creation when color scheme doesn't match."""
        nvtx_event = ChromeTraceEvent(
            name="forward",
            ph="X",
            cat="nvtx",
            ts=100.0,
            dur=50.0,
            pid="CPU",
            tid="1",
            args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
        )
        options = ConversionOptions(
            nvtx_color_scheme={"compute": "thread_state_running"}
        )
        
        event = _create_nvtx_kernel_event(
            nvtx_event, 120000, 150000, 0, options
        )
        
        # Should not have cname if no match
        self.assertIsNone(event.cname)


class TestProcessDeviceNvtxEvents(unittest.TestCase):
    """Test cases for _process_device_nvtx_events function."""

    def test_process_device_nvtx_events_successful_linking(self):
        """Test successful linking of NVTX events to kernels."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
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
        kernel_events = [
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
        ]
        
        adapter = NsysTraceEventAdapter()
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = _process_device_nvtx_events(
            nvtx_events, cuda_api_events, kernel_events, 0, adapter, options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 1)
        self.assertEqual(nvtx_kernel_events[0].name, "forward")
        self.assertEqual(nvtx_kernel_events[0].cat, "nvtx-kernel")
        self.assertEqual(len(mapped_identifiers), 1)
        self.assertGreater(len(flow_events), 0)

    def test_process_device_nvtx_events_no_overlap(self):
        """Test when NVTX events don't overlap with CUDA API events."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=200.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 200000, "end_ns": 210000}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel",
                ph="X",
                cat="kernel",
                ts=210.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 210000, "end_ns": 240000}
            ),
        ]
        
        adapter = NsysTraceEventAdapter()
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = _process_device_nvtx_events(
            nvtx_events, cuda_api_events, kernel_events, 0, adapter, options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 0)
        self.assertEqual(len(mapped_identifiers), 0)
        # Flow events should still be generated for CUDA API -> kernel links
        self.assertGreater(len(flow_events), 0)

    def test_process_device_nvtx_events_multiple_kernels(self):
        """Test linking with multiple kernels."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=100.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 200000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
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
        kernel_events = [
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 120000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="kernel2",
                ph="X",
                cat="kernel",
                ts=150.0,
                dur=40.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 150000, "end_ns": 190000}
            ),
        ]
        
        adapter = NsysTraceEventAdapter()
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = _process_device_nvtx_events(
            nvtx_events, cuda_api_events, kernel_events, 0, adapter, options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 1)
        # Duration should span from min start (120000) to max end (190000)
        self.assertEqual(nvtx_kernel_events[0].ts, 120.0)
        self.assertEqual(nvtx_kernel_events[0].dur, 70.0)  # (190000 - 120000) / 1000


class TestLinkNvtxToKernels(unittest.TestCase):
    """Test cases for link_nvtx_to_kernels function."""

    def test_link_nvtx_to_kernels_basic(self):
        """Test basic linking functionality."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
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
        kernel_events = [
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
        ]
        
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = link_nvtx_to_kernels(
            nvtx_events, cuda_api_events, kernel_events, options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 1)
        self.assertEqual(len(mapped_identifiers), 1)
        self.assertGreater(len(flow_events), 0)

    def test_link_nvtx_to_kernels_multiple_devices(self):
        """Test linking with multiple devices."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward_device0",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
            ),
            ChromeTraceEvent(
                name="forward_device1",
                ph="X",
                cat="nvtx",
                ts=200.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 1, "start_ns": 200000, "end_ns": 250000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel0",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 0",
                tid="1",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 110000, "end_ns": 120000}
            ),
            ChromeTraceEvent(
                name="cudaLaunchKernel1",
                ph="X",
                cat="cuda_api",
                ts=210.0,
                dur=10.0,
                pid="Device 1",
                tid="1",
                args={"deviceId": 1, "correlationId": 67890, "start_ns": 210000, "end_ns": 220000}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel0",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 0",
                tid="2",
                args={"deviceId": 0, "correlationId": 12345, "start_ns": 120000, "end_ns": 150000}
            ),
            ChromeTraceEvent(
                name="kernel1",
                ph="X",
                cat="kernel",
                ts=220.0,
                dur=30.0,
                pid="Device 1",
                tid="2",
                args={"deviceId": 1, "correlationId": 67890, "start_ns": 220000, "end_ns": 250000}
            ),
        ]
        
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = link_nvtx_to_kernels(
            nvtx_events, cuda_api_events, kernel_events, options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 2)
        self.assertEqual(len(mapped_identifiers), 2)

    def test_link_nvtx_to_kernels_no_common_devices(self):
        """Test when devices don't have all three event types."""
        nvtx_events = [
            ChromeTraceEvent(
                name="forward",
                ph="X",
                cat="nvtx",
                ts=100.0,
                dur=50.0,
                pid="CPU",
                tid="1",
                args={"deviceId": 0, "start_ns": 100000, "end_ns": 150000, "raw_tid": 1}
            ),
        ]
        cuda_api_events = [
            ChromeTraceEvent(
                name="cudaLaunchKernel",
                ph="X",
                cat="cuda_api",
                ts=110.0,
                dur=10.0,
                pid="Device 1",  # Different device
                tid="1",
                args={"deviceId": 1, "correlationId": 12345, "start_ns": 110000, "end_ns": 120000}
            ),
        ]
        kernel_events = [
            ChromeTraceEvent(
                name="kernel",
                ph="X",
                cat="kernel",
                ts=120.0,
                dur=30.0,
                pid="Device 1",  # Different device
                tid="2",
                args={"deviceId": 1, "correlationId": 12345, "start_ns": 120000, "end_ns": 150000}
            ),
        ]
        
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = link_nvtx_to_kernels(
            nvtx_events, cuda_api_events, kernel_events, options
        )
        
        # No common devices, so no linking
        self.assertEqual(len(nvtx_kernel_events), 0)
        self.assertEqual(len(mapped_identifiers), 0)

    def test_link_nvtx_to_kernels_empty_inputs(self):
        """Test with empty input lists."""
        options = ConversionOptions()
        
        nvtx_kernel_events, mapped_identifiers, flow_events = link_nvtx_to_kernels(
            [], [], [], options
        )
        
        self.assertEqual(len(nvtx_kernel_events), 0)
        self.assertEqual(len(mapped_identifiers), 0)
        self.assertEqual(len(flow_events), 0)

