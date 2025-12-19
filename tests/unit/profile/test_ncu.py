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
Tests for ncompass.profile.ncu module.

Tests the ncu integration functions for profiling kernels.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ncompass.profile.ncu import (
    check_ncu_available,
    query_ncu_metrics,
    filter_available_metrics,
    build_ncu_command,
    run_ncu_and_parse_output,
    get_metrics_str,
    run_ncu_profile,
)


class TestCheckNcuAvailable(unittest.TestCase):
    """Test cases for check_ncu_available function."""

    @patch("subprocess.run")
    def test_check_ncu_available_true(self, mock_run):
        """Test returns True when ncu is found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA (R) Nsight Compute Command Line Utility version 2023.1.1.0"
        )
        
        result = check_ncu_available()
        
        self.assertTrue(result)
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], ["ncu", "--version"])

    @patch("subprocess.run")
    def test_check_ncu_available_false_file_not_found(self, mock_run):
        """Test returns False when ncu is not found."""
        mock_run.side_effect = FileNotFoundError("ncu not found")
        
        result = check_ncu_available()
        
        self.assertFalse(result)

    @patch("subprocess.run")
    def test_check_ncu_available_false_called_process_error(self, mock_run):
        """Test returns False when ncu returns error."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ncu")
        
        result = check_ncu_available()
        
        self.assertFalse(result)


class TestQueryNcuMetrics(unittest.TestCase):
    """Test cases for query_ncu_metrics function."""

    @patch("subprocess.run")
    def test_query_ncu_metrics_success(self, mock_run):
        """Test querying metrics successfully."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="gpu__time_duration\nsm__cycles_elapsed\n"
        )
        
        metrics = query_ncu_metrics()
        
        self.assertEqual(metrics, {"gpu__time_duration", "sm__cycles_elapsed"})
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0], ["ncu", "--query-metrics"])

    @patch("subprocess.run")
    def test_query_ncu_metrics_failure(self, mock_run):
        """Test failure in querying metrics returns empty set."""
        mock_run.side_effect = FileNotFoundError()
        
        metrics = query_ncu_metrics()
        
        self.assertEqual(metrics, set())


class TestFilterAvailableMetrics(unittest.TestCase):
    """Test cases for filter_available_metrics function."""

    def test_filter_available_metrics(self):
        """Test filtering metrics based on available base metrics."""
        metrics_list = ["gpu__time_duration.sum", "missing_metric.avg", "sm__cycles_elapsed"]
        available_base_metrics = {"gpu__time_duration", "sm__cycles_elapsed"}
        
        filtered, missing = filter_available_metrics(metrics_list, available_base_metrics)
        
        self.assertEqual(filtered, ["gpu__time_duration.sum", "sm__cycles_elapsed"])
        self.assertEqual(missing, ["missing_metric.avg"])


class TestBuildNcuCommand(unittest.TestCase):
    """Test cases for build_ncu_command function."""

    def test_build_ncu_command_basic(self):
        """Test building basic NCU command."""
        cmd = build_ncu_command(
            ncu_bin="ncu",
            kernel_name="my_kernel",
            nvtx_include="my_range",
            metrics_str="metric1,metric2",
            command=["python", "script.py"]
        )
        
        expected = [
            "ncu",
            "--kernel-name", "my_kernel",
            "--target-processes", "all",
            "--nvtx",
            "--nvtx-include", "my_range",
            "--metrics", "metric1,metric2",
            "--csv",
            "--force-overwrite",
            "python", "script.py"
        ]
        self.assertEqual(cmd, expected)

    def test_build_ncu_command_no_nvtx(self):
        """Test building NCU command without nvtx filter."""
        cmd = build_ncu_command(
            ncu_bin="ncu",
            kernel_name="",
            nvtx_include="",
            metrics_str="metric1",
            command=["./app"]
        )
        
        self.assertNotIn("--nvtx-include", cmd)
        self.assertEqual(cmd[2], "") # kernel_name is empty


class TestRunNcuAndParseOutput(unittest.TestCase):
    """Test cases for run_ncu_and_parse_output function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.working_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_run_ncu_and_parse_output_success(self, mock_run):
        """Test successful run and parsing of CSV."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='Some random text\n"ID","Process ID","Other"\n"1","123","val"'
        )
        output_csv = self.working_dir / "output.csv"
        
        run_ncu_and_parse_output(["ncu", "args"], self.working_dir, output_csv)
        
        self.assertTrue(output_csv.exists())
        content = output_csv.read_text()
        self.assertIn('"ID","Process ID"', content)
        self.assertNotIn('Some random text', content)

    @patch("subprocess.run")
    def test_run_ncu_and_parse_output_no_csv(self, mock_run):
        """Test failure when no CSV data is found."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='No CSV data here'
        )
        output_csv = self.working_dir / "output.csv"
        
        with self.assertRaises(RuntimeError) as cm:
            run_ncu_and_parse_output(["ncu", "args"], self.working_dir, output_csv)
        self.assertIn("No valid CSV data found", str(cm.exception))


class TestGetMetricsStr(unittest.TestCase):
    """Test cases for get_metrics_str function."""

    @patch("ncompass.profile.ncu.query_ncu_metrics")
    def test_get_metrics_str_success(self, mock_query):
        """Test getting metrics string successfully."""
        mock_query.return_value = {"metric1", "metric2"}
        metrics_list = ["metric1.sum", "metric2.avg", "metric3.max"]
        
        result = get_metrics_str(metrics_list)
        
        self.assertEqual(result, "metric1.sum,metric2.avg")

    @patch("ncompass.profile.ncu.query_ncu_metrics")
    def test_get_metrics_str_no_valid_metrics(self, mock_query):
        """Test failure when no valid metrics are found."""
        mock_query.return_value = {"other"}
        metrics_list = ["metric1.sum"]
        
        with self.assertRaises(ValueError):
            get_metrics_str(metrics_list)


class TestRunNcuProfile(unittest.TestCase):
    """Test cases for run_ncu_profile function."""

    @patch("ncompass.profile.ncu.get_metrics_str")
    @patch("ncompass.profile.ncu.run_ncu_and_parse_output")
    def test_run_ncu_profile_success(self, mock_run, mock_get_metrics):
        """Test successful profiling run."""
        mock_get_metrics.return_value = "metric1,metric2"
        trace_dir = Path("/tmp/traces")
        
        with patch("pathlib.Path.mkdir"): # Avoid creating real dir
            result = run_ncu_profile(
                command=["python", "test.py"],
                output_name="test_out",
                trace_dir=trace_dir,
                working_dir=Path("/tmp")
            )
            
            self.assertEqual(result, trace_dir / "test_out.csv")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            self.assertIn("--metrics", cmd)
            self.assertIn("metric1,metric2", cmd)


if __name__ == "__main__":
    unittest.main()
