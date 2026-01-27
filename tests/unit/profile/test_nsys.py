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
    NsysDefaults,
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
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
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
        )
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        
        self.assertEqual(cmd[0], "nsys")
        self.assertEqual(cmd[1], "profile")
        self.assertIn("--gpuctxsw=true", cmd)
        self.assertIn("--cuda-graph-trace=node", cmd)
        self.assertIn("--force-overwrite=true", cmd)
        self.assertIn("--stop-on-exit=true", cmd)
        
        self.assertIn(str(self.script_path), cmd)
        self.assertIn("--arg1", cmd)
        self.assertIn("value1", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_with_extra_args(self, mock_run):
        """Test extra_args can override defaults."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            extra_args=["--trace", "cuda,nvtx", "--sample", "none"],
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "nsys")
        self.assertEqual(cmd[1], "profile")
        self.assertIn("--trace=cuda,nvtx", cmd)
        self.assertIn("--sample=none", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_with_capture_range(self, mock_run):
        """Test cudaProfilerApi capture range is included by default."""
        mock_run.return_value = MagicMock(returncode=0)

        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()

        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
        )

        cmd = mock_run.call_args[0][0]
        self.assertIn("--capture-range=cudaProfilerApi", cmd)
        self.assertIn("--capture-range-end=repeat", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_default_trace_types(self, mock_run):
        """Test default trace types are included."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
        )
        
        cmd = mock_run.call_args[0][0]
        trace_arg = [arg for arg in cmd if arg.startswith("--trace=")]
        self.assertEqual(len(trace_arg), 1)
        self.assertIn("cuda", trace_arg[0])
        self.assertIn("nvtx", trace_arg[0])

    @patch("subprocess.run")
    def test_run_nsys_profile_override_via_extra_args(self, mock_run):
        """Test extra_args can add custom options."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            extra_args=["--python-backtrace", "cuda", "--pytorch", "functions-trace"],
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--python-backtrace=cuda", cmd)
        self.assertIn("--pytorch=functions-trace", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_force_overwrite(self, mock_run):
        """Test force_overwrite flag is included by default."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--force-overwrite=true", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_no_force_overwrite(self, mock_run):
        """Test force_overwrite can be disabled via extra_args."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            extra_args=["--force-overwrite", "false"],
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--force-overwrite=false", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_output_path(self, mock_run):
        """Test output path is correctly set in command."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
        )
        
        cmd = mock_run.call_args[0][0]
        output_arg = [arg for arg in cmd if arg.startswith("--output=")]
        self.assertEqual(len(output_arg), 1)
        self.assertIn("test_output", output_arg[0])


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
        )
        
        self.assertIsNone(result)

    @patch("subprocess.run")
    def test_run_nsys_profile_output_not_found(self, mock_run):
        """Test returns None when output file is not created."""
        mock_run.return_value = MagicMock(returncode=0)
        
        result = run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
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
        )
        
        self.assertEqual(result, expected_output)
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[-1], str(self.script_path))

    @patch("subprocess.run")
    def test_run_nsys_profile_gpuctxsw_false(self, mock_run):
        """Test gpuctxsw can be overridden to false."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            extra_args=["--gpuctxsw", "false"],
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--gpuctxsw=false", cmd)

    @patch("subprocess.run")
    def test_run_nsys_profile_default_sample(self, mock_run):
        """Test default sample mode is included."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--sample=process-tree", cmd)

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
        )
        
        call_kwargs = mock_run.call_args[1]
        self.assertTrue(call_kwargs["check"])

    @patch("subprocess.run")
    def test_run_nsys_profile_cuda_graph_trace_graph_mode(self, mock_run):
        """Test cuda_graph_trace can be overridden to 'graph'."""
        mock_run.return_value = MagicMock(returncode=0)
        
        expected_output = self.trace_dir / "test_output.nsys-rep"
        expected_output.touch()
        
        run_nsys_profile(
            command=[str(self.script_path)],
            output_name="test_output",
            trace_dir=self.trace_dir,
            working_dir=self.script_path.parent,
            extra_args=["--cuda-graph-trace", "graph"],
        )
        
        cmd = mock_run.call_args[0][0]
        self.assertIn("--cuda-graph-trace=graph", cmd)


class TestNsysDefaults(unittest.TestCase):
    """Test cases for NsysDefaults class."""

    def test_to_dict_returns_nsys_argument_format(self):
        """Test that to_dict returns dictionary with nsys argument keys."""
        defaults = NsysDefaults()
        d = defaults.to_dict()

        # Verify keys are in nsys --key format
        for key in d.keys():
            self.assertTrue(key.startswith("--"), f"Key should start with '--': {key}")

    def test_to_dict_contains_expected_keys(self):
        """Test that to_dict contains all expected nsys arguments."""
        defaults = NsysDefaults()
        d = defaults.to_dict()

        expected_keys = [
            "--trace",
            "--sample",
            "--gpuctxsw",
            "--cuda-graph-trace",
            "--stop-on-exit",
            "--trace-fork-before-exec",
            "--force-overwrite",
            "--capture-range",
            "--capture-range-end",
        ]

        for key in expected_keys:
            self.assertIn(key, d, f"Missing expected key: {key}")

    def test_to_dict_capture_range_is_cuda_profiler_api(self):
        """Test that capture_range defaults to cudaProfilerApi."""
        defaults = NsysDefaults()
        d = defaults.to_dict()

        self.assertEqual(d["--capture-range"], "cudaProfilerApi")
        self.assertEqual(d["--capture-range-end"], "repeat")

    def test_to_dict_no_nvtx_capture(self):
        """Test that to_dict does not contain nvtx-capture key."""
        defaults = NsysDefaults()
        d = defaults.to_dict()

        # The new NsysDefaults uses cudaProfilerApi capture range instead of nvtx
        # So nvtx-capture should not be present
        self.assertNotIn("--nvtx-capture", d)

    def test_to_dict_default_trace_types(self):
        """Test that default trace types include cuda and nvtx."""
        defaults = NsysDefaults()
        d = defaults.to_dict()

        trace_value = d["--trace"]
        self.assertIn("cuda", trace_value)
        self.assertIn("nvtx", trace_value)


if __name__ == "__main__":
    unittest.main()
