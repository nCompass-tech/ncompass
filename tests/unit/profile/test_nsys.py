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
Tests for ncompass.profile.nsys module.

Tests the nsys integration functions for profiling Python scripts.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ncompass.profile.nsys import (
    check_nsys_available,
    create_trace_directory,
    run_nsys_profile,
)


class TestCheckNsysAvailable(unittest.TestCase):
    """Test cases for check_nsys_available function."""

    @patch("subprocess.run")
    def test_check_nsys_available_true(self, mock_run):
        """Test returns True when nsys is found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA Nsight Systems version 2023.4.1.97-234519059v0"
        )
        
        result = check_nsys_available()
        
        self.assertTrue(result)
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], ["nsys", "--version"])

    @patch("subprocess.run")
    def test_check_nsys_available_false_file_not_found(self, mock_run):
        """Test returns False when nsys is not found."""
        mock_run.side_effect = FileNotFoundError("nsys not found")
        
        result = check_nsys_available()
        
        self.assertFalse(result)

    @patch("subprocess.run")
    def test_check_nsys_available_false_called_process_error(self, mock_run):
        """Test returns False when nsys returns error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "nsys")
        
        result = check_nsys_available()
        
        self.assertFalse(result)


class TestCreateTraceDirectory(unittest.TestCase):
    """Test cases for create_trace_directory function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_trace_directory_creates_dir(self):
        """Test that directory is created."""
        base_dir = Path(self.temp_dir)
        
        trace_dir, timestamp = create_trace_directory(base_dir)
        
        self.assertTrue(trace_dir.exists())
        self.assertTrue(trace_dir.is_dir())

    def test_create_trace_directory_returns_tuple(self):
        """Test that function returns (Path, str) tuple."""
        base_dir = Path(self.temp_dir)
        
        result = create_trace_directory(base_dir)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Path)
        self.assertIsInstance(result[1], str)

    def test_create_trace_directory_nested(self):
        """Test that .nsys_traces subdirectory is created."""
        base_dir = Path(self.temp_dir)
        
        trace_dir, _ = create_trace_directory(base_dir)
        
        # Should be base_dir/.nsys_traces/<timestamp>
        self.assertEqual(trace_dir.parent.parent, base_dir)
        self.assertEqual(trace_dir.parent.name, ".nsys_traces")

    def test_create_trace_directory_timestamp_format(self):
        """Test timestamp follows expected format YYYYMMDD_HHMMSS."""
        base_dir = Path(self.temp_dir)
        
        trace_dir, timestamp = create_trace_directory(base_dir)
        
        # Timestamp should be 15 characters: YYYYMMDD_HHMMSS
        self.assertEqual(len(timestamp), 15)
        self.assertEqual(timestamp[8], "_")
        # All other chars should be digits
        self.assertTrue(timestamp[:8].isdigit())
        self.assertTrue(timestamp[9:].isdigit())

    def test_create_trace_directory_timestamp_matches_dir_name(self):
        """Test timestamp matches directory name."""
        base_dir = Path(self.temp_dir)
        
        trace_dir, timestamp = create_trace_directory(base_dir)
        
        self.assertEqual(trace_dir.name, timestamp)


class TestRunNsysProfileSuccess(unittest.TestCase):
    """Positive test cases for run_nsys_profile function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_path = Path(self.temp_dir) / "test_script.py"
        self.script_path.write_text("print('hello')")
        self.trace_dir = Path(self.temp_dir) / "traces"
        self.trace_dir.mkdir()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_run_nsys_profile_success(self, mock_run):
        """Test successful profiling returns path to nsys-rep file."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Create the expected output file
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda,nvtx",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=True,
            use_sudo=False,
            cache_dir=None,
        )
        
        self.assertEqual(result, expected_output)

    @patch("subprocess.run")
    def test_run_nsys_profile_builds_correct_command(self, mock_run):
        """Test that correct nsys command is built."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path), "--arg1", "value1"],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda,nvtx,osrt",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=True,
            use_sudo=False,
            cache_dir=None,
        )
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        
        # Verify command structure
        self.assertEqual(cmd[0], "nsys")
        self.assertEqual(cmd[1], "profile")
        self.assertIn("--trace=cuda,nvtx,osrt", cmd)
        self.assertIn("--sample=process-tree", cmd)
        self.assertIn("--session-new=nc0", cmd)
        self.assertIn("--gpuctxsw=true", cmd)
        self.assertIn("--cuda-graph-trace=node", cmd)
        self.assertIn("--cuda-memory-usage=true", cmd)
        self.assertIn("--force-overwrite=true", cmd)
        self.assertIn("--show-output=true", cmd)
        self.assertIn("--stop-on-exit=true", cmd)
        
        # Verify script and args are at the end
        self.assertIn(str(self.script_path), cmd)
        self.assertIn("--arg1", cmd)
        self.assertIn("value1", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_with_sudo(self, mock_run):
        """Test sudo -E is prepended when use_sudo=True."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=True,
            use_sudo=True,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "sudo")
        self.assertEqual(cmd[1], "-E")
        self.assertEqual(cmd[2], "nsys")

    @patch("subprocess.run")
    def test_run_nsys_profile_with_range(self, mock_run):
        """Test NVTX range capture options are added when with_range=True."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=True,
            python_tracing=True,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--capture-range=nvtx", cmd)
        self.assertIn("--nvtx-capture=nc_start_capture", cmd)
        self.assertIn("--capture-range-end=repeat", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_python_tracing(self, mock_run):
        """Test Python/PyTorch tracing options are added when python_tracing=True."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=True,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--cudabacktrace=kernel", cmd)
        self.assertIn("--python-backtrace=cuda", cmd)
        self.assertIn("--pytorch=functions-trace", cmd)
        self.assertIn("--python-sampling=true", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_no_python_tracing(self, mock_run):
        """Test Python tracing options are not added when python_tracing=False."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--cudabacktrace=kernel", cmd)
        self.assertNotIn("--python-backtrace=cuda", cmd)
        self.assertNotIn("--pytorch=functions-trace", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_force_overwrite(self, mock_run):
        """Test force_overwrite flag is added correctly."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        # With force_overwrite=True
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--force-overwrite=true", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_no_force_overwrite(self, mock_run):
        """Test force_overwrite flag is not added when False."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=False,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--force-overwrite=true", cmd)

    @patch.dict(os.environ, {}, clear=False)
    @patch("subprocess.run")
    def test_run_nsys_profile_cache_dir_env(self, mock_run):
        """Test cache_dir sets NCOMPASS_CACHE_DIR environment variable."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        cache_dir = "/tmp/ncompass_cache"
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=cache_dir,
        )
        
        # Check that the environment variable was set
        self.assertEqual(os.environ.get("NCOMPASS_CACHE_DIR"), cache_dir)


class TestRunNsysProfileNegative(unittest.TestCase):
    """Negative test cases for run_nsys_profile function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_path = Path(self.temp_dir) / "test_script.py"
        self.script_path.write_text("print('hello')")
        self.trace_dir = Path(self.temp_dir) / "traces"
        self.trace_dir.mkdir()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_run_nsys_profile_subprocess_fails(self, mock_run):
        """Test returns None when subprocess fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "nsys")
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        self.assertIsNone(result)

    @patch("subprocess.run")
    def test_run_nsys_profile_output_not_found(self, mock_run):
        """Test returns None when output file is not created."""
        mock_run.return_value = MagicMock(returncode=0)
        
        # Don't create the expected output file
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        self.assertIsNone(result)


class TestRunNsysProfileEdgeCases(unittest.TestCase):
    """Edge case tests for run_nsys_profile function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.script_path = Path(self.temp_dir) / "test_script.py"
        self.script_path.write_text("print('hello')")
        self.trace_dir = Path(self.temp_dir) / "traces"
        self.trace_dir.mkdir()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("subprocess.run")
    def test_run_nsys_profile_empty_command_args(self, mock_run):
        """Test handles command with no extra args correctly."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        self.assertEqual(result, expected_output)
        cmd = mock_run.call_args[0][0]
        # Script path should be at the end, no extra args
        self.assertEqual(cmd[-1], str(self.script_path))

    @patch("subprocess.run")
    def test_run_nsys_profile_gpuctxsw_false(self, mock_run):
        """Test gpuctxsw=false is in command."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=False,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--gpuctxsw=false", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_cuda_memory_usage_false(self, mock_run):
        """Test cuda-memory-usage=false is in command."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=False,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--cuda-memory-usage=false", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_cwd_is_working_dir(self, mock_run):
        """Test subprocess is run with cwd set to working_dir."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["cwd"], self.script_path.parent)

    @patch("subprocess.run")
    def test_run_nsys_profile_check_is_true(self, mock_run):
        """Test subprocess.run is called with check=True."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="node",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        call_kwargs = mock_run.call_args[1]
        self.assertTrue(call_kwargs["check"])

    @patch("subprocess.run")
    def test_run_nsys_profile_cuda_graph_trace_graph_mode(self, mock_run):
        """Test cuda_graph_trace='graph' is in command."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            trace_types="cuda",
            force_overwrite=True,
            sample="process-tree",
            session_name="nc0",
            gpuctxsw=True,
            cuda_graph_trace="graph",
            cuda_memory_usage=True,
            with_range=False,
            python_tracing=False,
            use_sudo=False,
            cache_dir=None,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--cuda-graph-trace=graph", cmd)


if __name__ == "__main__":
    unittest.main()
