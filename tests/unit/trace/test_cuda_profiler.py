# Copyright 2025 nCompass Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for ncompass.trace.profile.cuda_profiler module.

Tests the CudaProfilerContext context manager for CUDA profiler start/stop.
"""

import sys
import unittest
from unittest.mock import MagicMock


# Mock torch before importing cuda_profiler module
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch


class TestCudaProfilerContext(unittest.TestCase):
    """Test cases for CudaProfilerContext class."""

    def setUp(self):
        """Reset mock before each test."""
        mock_torch.reset_mock()

    def test_init_default_name(self):
        """Test CudaProfilerContext initializes with empty name by default."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        ctx = CudaProfilerContext()
        self.assertEqual(ctx.name, "")

    def test_init_with_name(self):
        """Test CudaProfilerContext initializes with provided name."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        ctx = CudaProfilerContext(name="test_region")
        self.assertEqual(ctx.name, "test_region")

    def test_enter_starts_profiler(self):
        """Test __enter__ calls torch.cuda.profiler.start()."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        ctx = CudaProfilerContext()
        result = ctx.__enter__()

        mock_torch.cuda.profiler.start.assert_called_once()
        self.assertIs(result, ctx)

    def test_exit_stops_profiler(self):
        """Test __exit__ calls torch.cuda.profiler.stop()."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        ctx = CudaProfilerContext()
        ctx.__exit__(None, None, None)

        mock_torch.cuda.profiler.stop.assert_called_once()

    def test_context_manager_usage(self):
        """Test CudaProfilerContext works as a context manager."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        with CudaProfilerContext(name="test"):
            # Verify start was called when entering
            mock_torch.cuda.profiler.start.assert_called_once()
            mock_torch.cuda.profiler.stop.assert_not_called()

        # Verify stop was called when exiting
        mock_torch.cuda.profiler.stop.assert_called_once()

    def test_exit_with_exception(self):
        """Test __exit__ is called even when exception occurs."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        try:
            with CudaProfilerContext():
                raise ValueError("test error")
        except ValueError:
            pass

        # Verify stop was still called despite exception
        mock_torch.cuda.profiler.stop.assert_called_once()

    def test_exit_returns_none(self):
        """Test __exit__ returns None (doesn't suppress exceptions)."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        ctx = CudaProfilerContext()
        result = ctx.__exit__(ValueError, ValueError("test"), None)

        # None means exception is not suppressed
        self.assertIsNone(result)

    def test_inherits_from_profile_context_base(self):
        """Test CudaProfilerContext inherits from ProfileContextBase."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext
        from ncompass.trace.profile.base import ProfileContextBase

        ctx = CudaProfilerContext()
        self.assertIsInstance(ctx, ProfileContextBase)


class TestCudaProfilerContextIntegration(unittest.TestCase):
    """Integration-style tests for CudaProfilerContext."""

    def setUp(self):
        """Reset mock before each test."""
        mock_torch.reset_mock()

    def test_multiple_context_managers(self):
        """Test multiple CudaProfilerContext instances work independently."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        with CudaProfilerContext(name="outer"):
            self.assertEqual(mock_torch.cuda.profiler.start.call_count, 1)

            with CudaProfilerContext(name="inner"):
                self.assertEqual(mock_torch.cuda.profiler.start.call_count, 2)

            self.assertEqual(mock_torch.cuda.profiler.stop.call_count, 1)

        self.assertEqual(mock_torch.cuda.profiler.stop.call_count, 2)

    def test_sequential_context_managers(self):
        """Test sequential CudaProfilerContext usage."""
        from ncompass.trace.profile.cuda_profiler import CudaProfilerContext

        with CudaProfilerContext(name="first"):
            pass

        with CudaProfilerContext(name="second"):
            pass

        self.assertEqual(mock_torch.cuda.profiler.start.call_count, 2)
        self.assertEqual(mock_torch.cuda.profiler.stop.call_count, 2)


if __name__ == "__main__":
    unittest.main()
