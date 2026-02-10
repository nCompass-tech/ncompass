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

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ncompass.profile.ncu import (
    check_ncu_available,
    query_ncu_metrics,
    filter_available_metrics,
    _build_ncu_command,
    convert_ncu_to_csv,
    convert_ncu_to_session,
    convert_ncu_to_source,
    get_metrics_str,
    run_ncu_profile,
    _parse_ncu_args,
    NcuDefaults,
    load_ncu_kernel_targets,
    build_kernel_id_regex,
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
    """Test cases for _build_ncu_command function."""

    def test_build_ncu_command_basic(self):
        """Test building basic NCU command."""
        output_path = Path("/tmp/test_out")
        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1,metric2",
            extra_args=[],
            command=["python", "script.py"]
        )
        
        # Check basic structure
        self.assertEqual(cmd[0], "ncu")
        self.assertIn("--nvtx", cmd)
        self.assertIn("--force-overwrite", cmd)
        
        # Check key=value pairs
        self.assertIn(f"--export={output_path}", cmd)
        self.assertIn("--metrics=metric1,metric2", cmd)
        self.assertIn("--target-processes=all", cmd)
        
        # Check command at the end
        self.assertEqual(cmd[-2:], ["python", "script.py"])

    def test_build_ncu_command_with_extra_args(self):
        """Test building NCU command with extra args overriding defaults."""
        output_path = Path("/tmp/test_out")
        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1",
            extra_args=["--target-processes", "none", "--clock-control=base"],
            command=["./app"]
        )

        self.assertIn("--target-processes=none", cmd)
        self.assertIn("--clock-control=base", cmd)
        self.assertNotIn("--target-processes=all", cmd)


class TestConvertNcuToCsv(unittest.TestCase):
    """Test cases for convert_ncu_to_csv function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.working_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_convert_ncu_to_csv_success(self, mock_run):
        """Test successful conversion of .ncu-rep to CSV."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='Some random text\n"ID","Process ID","Other"\n"1","123","val"'
        )
        ncu_rep = self.working_dir / "test.ncu-rep"
        output_csv = self.working_dir / "output.csv"
        
        convert_ncu_to_csv(ncu_rep, output_csv)
        
        self.assertTrue(output_csv.exists())
        content = output_csv.read_text()
        self.assertIn('"ID","Process ID"', content)
        self.assertNotIn('Some random text', content)

    @patch("subprocess.run")
    def test_convert_ncu_to_csv_no_csv_data(self, mock_run):
        """Test failure when no CSV data is found in ncu output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='No CSV data here'
        )
        ncu_rep = self.working_dir / "test.ncu-rep"
        output_csv = self.working_dir / "output.csv"
        
        with self.assertRaises(RuntimeError) as cm:
            convert_ncu_to_csv(ncu_rep, output_csv)
        self.assertIn("No valid CSV data found", str(cm.exception))


class TestConvertNcuToSession(unittest.TestCase):
    """Test cases for convert_ncu_to_session function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.working_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_convert_ncu_to_session_success(self, mock_run):
        """Test successful export of session info."""
        session_text = "Device: NVIDIA H100\nCompute Capability: 9.0\n"
        mock_run.return_value = MagicMock(returncode=0, stdout=session_text)

        ncu_rep = self.working_dir / "test.ncu-rep"
        output = self.working_dir / "test.session"

        convert_ncu_to_session(ncu_rep, output)

        self.assertTrue(output.exists())
        self.assertEqual(output.read_text(), session_text)

        call_args = mock_run.call_args[0][0]
        self.assertEqual(call_args, ["ncu", "--import", str(ncu_rep), "--page", "session"])

    @patch("subprocess.run")
    def test_convert_ncu_to_session_failure(self, mock_run):
        """Test RuntimeError raised on subprocess failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ncu", stderr="error")

        ncu_rep = self.working_dir / "test.ncu-rep"
        output = self.working_dir / "test.session"

        with self.assertRaises(RuntimeError) as cm:
            convert_ncu_to_session(ncu_rep, output)
        self.assertIn("session export failed", str(cm.exception))


class TestConvertNcuToSource(unittest.TestCase):
    """Test cases for convert_ncu_to_source function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.working_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("subprocess.run")
    def test_convert_ncu_to_source_sass(self, mock_run):
        """Test successful export of SASS source."""
        sass_text = "IMAD.MOV R1, RZ, RZ, c[0x0][0x28]\n"
        mock_run.return_value = MagicMock(returncode=0, stdout=sass_text)

        ncu_rep = self.working_dir / "test.ncu-rep"
        output = self.working_dir / "test.source.sass"

        convert_ncu_to_source(ncu_rep, output, "sass")

        self.assertTrue(output.exists())
        self.assertEqual(output.read_text(), sass_text)

        call_args = mock_run.call_args[0][0]
        self.assertEqual(
            call_args,
            ["ncu", "--import", str(ncu_rep), "--page", "source", "--print-source", "sass"],
        )

    @patch("subprocess.run")
    def test_convert_ncu_to_source_ptx(self, mock_run):
        """Test successful export of PTX source."""
        ptx_text = "mov.u32 %r1, %ctaid.x;\n"
        mock_run.return_value = MagicMock(returncode=0, stdout=ptx_text)

        ncu_rep = self.working_dir / "test.ncu-rep"
        output = self.working_dir / "test.source.ptx"

        convert_ncu_to_source(ncu_rep, output, "ptx")

        self.assertTrue(output.exists())
        self.assertEqual(output.read_text(), ptx_text)

        call_args = mock_run.call_args[0][0]
        self.assertEqual(
            call_args,
            ["ncu", "--import", str(ncu_rep), "--page", "source", "--print-source", "ptx"],
        )

    @patch("subprocess.run")
    def test_convert_ncu_to_source_failure(self, mock_run):
        """Test RuntimeError raised on subprocess failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "ncu", stderr="error")

        ncu_rep = self.working_dir / "test.ncu-rep"
        output = self.working_dir / "test.source.sass"

        with self.assertRaises(RuntimeError) as cm:
            convert_ncu_to_source(ncu_rep, output, "sass")
        self.assertIn("source (sass) export failed", str(cm.exception))


class TestParseNcuArgs(unittest.TestCase):
    """Test cases for _parse_ncu_args function."""

    def test_parse_ncu_args_mixed_formats(self):
        """Test parsing arguments with different formats."""
        args = ["--key1=val1", "--key2", "val2", "--flag"]
        parsed = _parse_ncu_args(args)
        
        self.assertEqual(parsed["--key1"], "val1")
        self.assertEqual(parsed["--key2"], "val2")
        self.assertEqual(parsed["--flag"], "true")


class TestNcuDefaults(unittest.TestCase):
    """Test cases for NcuDefaults class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        defaults = NcuDefaults()
        d = defaults.to_dict()

        self.assertEqual(d["--target-processes"], "all")
        self.assertEqual(d["--nvtx-include"], "regex:user_annotated:.*/")
        self.assertEqual(d["--clock-control"], "none")


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
    @patch("subprocess.run")
    def test_run_ncu_profile_failure(self, mock_run, mock_get_metrics):
        """Test profiling run failure."""
        mock_get_metrics.return_value = "metric1"
        mock_run.side_effect = subprocess.CalledProcessError(1, "ncu")
        trace_dir = Path("/tmp/traces")
        
        result = run_ncu_profile(
            command=["python", "test.py"],
            output_name="test_out",
            trace_dir=trace_dir
        )
        
        self.assertIsNone(result)


class TestLoadNcuKernelTargets(unittest.TestCase):
    """Test cases for load_ncu_kernel_targets function."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_targets_file(self, targets: list, trace_file_name: str = "test_trace"):
        """Helper to create config.json file for a trace."""
        import json
        targets_dir = self.cache_dir / ".cache" / "ncompass" / "profiles" / ".default" / "NCU" / trace_file_name / "current"
        targets_dir.mkdir(parents=True, exist_ok=True)
        targets_path = targets_dir / "config.json"
        with open(targets_path, 'w') as f:
            json.dump({"targets": targets}, f)
        return targets_path

    def test_load_ncu_kernel_targets_no_file(self):
        """Test returns empty list when no targets file exists."""
        result = load_ncu_kernel_targets(self.cache_dir, "nonexistent_trace")
        self.assertEqual(result, [])

    def test_load_ncu_kernel_targets_with_targets(self):
        """Test loading kernel targets from file for specific trace."""
        targets = [
            {"kernel_name": "kernel_A", "instance_number": 3},
            {"kernel_name": "kernel_B", "instance_number": 1}
        ]
        self._create_targets_file(targets, "my_trace")

        result = load_ncu_kernel_targets(self.cache_dir, "my_trace")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["kernel_name"], "kernel_A")
        self.assertEqual(result[0]["instance_number"], 3)
        self.assertEqual(result[1]["kernel_name"], "kernel_B")
        self.assertEqual(result[1]["instance_number"], 1)

    def test_load_ncu_kernel_targets_all_traces(self):
        """Test loading kernel targets from all trace files."""
        self._create_targets_file([{"kernel_name": "kernel_A", "instance_number": 1}], "trace_one")
        self._create_targets_file([{"kernel_name": "kernel_B", "instance_number": 2}], "trace_two")

        # Pass None to get all traces
        result = load_ncu_kernel_targets(self.cache_dir, None)

        self.assertEqual(len(result), 2)
        kernel_names = [t["kernel_name"] for t in result]
        self.assertIn("kernel_A", kernel_names)
        self.assertIn("kernel_B", kernel_names)

    def test_load_ncu_kernel_targets_empty_list(self):
        """Test loading empty targets list."""
        self._create_targets_file([], "empty_trace")

        result = load_ncu_kernel_targets(self.cache_dir, "empty_trace")

        self.assertEqual(result, [])

    def test_load_ncu_kernel_targets_invalid_json(self):
        """Test returns empty list for invalid JSON."""
        targets_dir = self.cache_dir / ".cache" / "ncompass" / "profiles" / ".default" / "NCU" / "bad_trace" / "current"
        targets_dir.mkdir(parents=True, exist_ok=True)
        targets_path = targets_dir / "config.json"
        targets_path.write_text("invalid json")

        result = load_ncu_kernel_targets(self.cache_dir, "bad_trace")

        self.assertEqual(result, [])


class TestBuildKernelIdRegex(unittest.TestCase):
    """Test cases for build_kernel_id_regex function."""

    def test_build_kernel_id_regex_single_target(self):
        """Test regex with single kernel target."""
        targets = [{"kernel_name": "gemm_kernel", "instance_number": 3}]
        result = build_kernel_id_regex(targets)
        self.assertEqual(result, "::regex:^(gemm_kernel)$:(3)")

    def test_build_kernel_id_regex_multiple_targets(self):
        """Test regex with multiple kernel targets - cross product."""
        targets = [
            {"kernel_name": "kernel_A", "instance_number": 1},
            {"kernel_name": "kernel_B", "instance_number": 5},
        ]
        result = build_kernel_id_regex(targets)
        # Kernel names sorted alphabetically, instances sorted numerically
        self.assertEqual(result, "::regex:^(kernel_A|kernel_B)$:(1|5)")

    def test_build_kernel_id_regex_same_kernel_multiple_instances(self):
        """Test regex with same kernel, different instances."""
        targets = [
            {"kernel_name": "gemm", "instance_number": 1},
            {"kernel_name": "gemm", "instance_number": 3},
            {"kernel_name": "gemm", "instance_number": 2},
        ]
        result = build_kernel_id_regex(targets)
        # Instances sorted numerically
        self.assertEqual(result, "::regex:^(gemm)$:(1|2|3)")

    def test_build_kernel_id_regex_mixed_targets(self):
        """Test regex with multiple kernels and instances."""
        targets = [
            {"kernel_name": "conv", "instance_number": 2},
            {"kernel_name": "gemm", "instance_number": 1},
            {"kernel_name": "gemm", "instance_number": 3},
            {"kernel_name": "conv", "instance_number": 5},
        ]
        result = build_kernel_id_regex(targets)
        # Cross product: (conv|gemm) x (1|2|3|5)
        self.assertEqual(result, "::regex:^(conv|gemm)$:(1|2|3|5)")

    def test_build_kernel_id_regex_empty_list(self):
        """Test regex with empty targets list returns None."""
        result = build_kernel_id_regex([])
        self.assertIsNone(result)

    def test_build_kernel_id_regex_none(self):
        """Test regex with None returns None."""
        result = build_kernel_id_regex(None)
        self.assertIsNone(result)

    def test_build_kernel_id_regex_missing_kernel_name(self):
        """Test regex skips targets with missing kernel_name."""
        targets = [
            {"kernel_name": "", "instance_number": 1},
            {"kernel_name": "valid_kernel", "instance_number": 2},
        ]
        result = build_kernel_id_regex(targets)
        self.assertEqual(result, "::regex:^(valid_kernel)$:(2)")


class TestBuildNcuCommandWithKernelTargets(unittest.TestCase):
    """Test cases for _build_ncu_command with kernel_targets parameter."""

    def test_build_ncu_command_with_kernel_targets(self):
        """Test building NCU command with kernel targets uses regex."""
        output_path = Path("/tmp/test_out")
        kernel_targets = [
            {"kernel_name": "gemm_kernel", "instance_number": 3}
        ]

        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1,metric2",
            extra_args=[],
            command=["python", "script.py"],
            kernel_targets=kernel_targets
        )

        # Check --kernel-id flag is present with regex format
        kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id=")]
        self.assertEqual(len(kernel_id_args), 1)
        self.assertIn("::regex:^(gemm_kernel)$:(3)", kernel_id_args[0])

    def test_build_ncu_command_with_multiple_kernel_targets(self):
        """Test building NCU command with multiple kernel targets uses single regex."""
        output_path = Path("/tmp/test_out")
        kernel_targets = [
            {"kernel_name": "kernel_A", "instance_number": 1},
            {"kernel_name": "kernel_B", "instance_number": 5}
        ]

        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1",
            extra_args=[],
            command=["./app"],
            kernel_targets=kernel_targets
        )

        # Check single --kernel-id flag with cross-product regex
        kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id=")]
        self.assertEqual(len(kernel_id_args), 1)
        self.assertIn("::regex:^(kernel_A|kernel_B)$:(1|5)", kernel_id_args[0])

    def test_build_ncu_command_no_kernel_targets(self):
        """Test building NCU command without kernel targets."""
        output_path = Path("/tmp/test_out")

        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1",
            extra_args=[],
            command=["./app"],
            kernel_targets=None
        )

        # No --kernel-id flags should be present
        kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id")]
        self.assertEqual(len(kernel_id_args), 0)

    def test_build_ncu_command_empty_kernel_targets(self):
        """Test building NCU command with empty kernel targets list."""
        output_path = Path("/tmp/test_out")

        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1",
            extra_args=[],
            command=["./app"],
            kernel_targets=[]
        )

        # No --kernel-id flags should be present
        kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id")]
        self.assertEqual(len(kernel_id_args), 0)

    def test_build_ncu_command_kernel_id_before_other_args(self):
        """Test that --kernel-id flags appear before key=value args."""
        output_path = Path("/tmp/test_out")
        kernel_targets = [{"kernel_name": "test_kernel", "instance_number": 1}]

        cmd = _build_ncu_command(
            output_path=output_path,
            metrics_str="metric1",
            extra_args=[],
            command=["./app"],
            kernel_targets=kernel_targets
        )

        # Find positions
        kernel_id_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("--kernel-id="))
        export_idx = next(i for i, arg in enumerate(cmd) if arg.startswith("--export="))

        # --kernel-id should appear before --export
        self.assertLess(kernel_id_idx, export_idx)


class TestRunNcuProfileWithKernelTargets(unittest.TestCase):
    """Test cases for run_ncu_profile with kernel targets."""

    @patch("ncompass.profile.ncu.load_ncu_kernel_targets")
    @patch("ncompass.profile.ncu.get_metrics_str")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_ncu_profile_loads_kernel_targets(self, mock_exists, mock_run, mock_get_metrics, mock_load_targets):
        """Test that run_ncu_profile loads and uses kernel targets with regex."""
        mock_get_metrics.return_value = "metric1"
        mock_exists.return_value = True
        mock_load_targets.return_value = [{"kernel_name": "my_kernel", "instance_number": 2}]
        trace_dir = Path("/tmp/traces")

        with patch("ncompass.profile.ncu.config") as mock_config:
            mock_config.ncu_metrics = ("metric1",)

            run_ncu_profile(
                command=["python", "test.py"],
                output_name="test_out",
                trace_dir=trace_dir,
                working_dir=Path("/tmp"),
                use_kernel_targets=True
            )

            mock_load_targets.assert_called_once_with(Path("/tmp"))
            cmd = mock_run.call_args[0][0]
            # Check for regex format
            kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id=")]
            self.assertEqual(len(kernel_id_args), 1)
            self.assertIn("::regex:^(my_kernel)$:(2)", kernel_id_args[0])

    @patch("ncompass.profile.ncu.load_ncu_kernel_targets")
    @patch("ncompass.profile.ncu.get_metrics_str")
    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_ncu_profile_skips_kernel_targets_when_disabled(self, mock_exists, mock_run, mock_get_metrics, mock_load_targets):
        """Test that run_ncu_profile skips kernel targets when use_kernel_targets=False."""
        mock_get_metrics.return_value = "metric1"
        mock_exists.return_value = True
        trace_dir = Path("/tmp/traces")

        with patch("ncompass.profile.ncu.config") as mock_config:
            mock_config.ncu_metrics = ("metric1",)

            run_ncu_profile(
                command=["python", "test.py"],
                output_name="test_out",
                trace_dir=trace_dir,
                use_kernel_targets=False
            )

            mock_load_targets.assert_not_called()
            cmd = mock_run.call_args[0][0]
            kernel_id_args = [arg for arg in cmd if arg.startswith("--kernel-id")]
            self.assertEqual(len(kernel_id_args), 0)


class TestLoadNcuKernelTargetsWithEnvVar(unittest.TestCase):
    """Test cases for load_ncu_kernel_targets with NCOMPASS_TRACE_NAME env var."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_dir.name)
        # Clear env vars before each test
        self.orig_trace_name = os.environ.get("NCOMPASS_TRACE_NAME")
        if "NCOMPASS_TRACE_NAME" in os.environ:
            del os.environ["NCOMPASS_TRACE_NAME"]

    def tearDown(self):
        self.temp_dir.cleanup()
        # Restore original env var
        if self.orig_trace_name is not None:
            os.environ["NCOMPASS_TRACE_NAME"] = self.orig_trace_name
        elif "NCOMPASS_TRACE_NAME" in os.environ:
            del os.environ["NCOMPASS_TRACE_NAME"]

    def _create_targets_file(self, targets: list, trace_file_name: str):
        """Helper to create config.json file for a trace."""
        import json
        targets_dir = self.cache_dir / ".cache" / "ncompass" / "profiles" / ".default" / "NCU" / trace_file_name / "current"
        targets_dir.mkdir(parents=True, exist_ok=True)
        targets_path = targets_dir / "config.json"
        with open(targets_path, 'w') as f:
            json.dump({"targets": targets}, f)
        return targets_path

    def test_load_with_env_var(self):
        """Test loading targets using NCOMPASS_TRACE_NAME env var."""
        targets = [{"kernel_name": "env_kernel", "instance_number": 7}]
        self._create_targets_file(targets, "env_trace")

        os.environ["NCOMPASS_TRACE_NAME"] = "env_trace"

        # Pass None for trace_file_name - should use env var
        result = load_ncu_kernel_targets(self.cache_dir, None)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["kernel_name"], "env_kernel")
        self.assertEqual(result[0]["instance_number"], 7)

    def test_explicit_trace_name_overrides_env_var(self):
        """Test that explicit trace_file_name overrides NCOMPASS_TRACE_NAME."""
        self._create_targets_file([{"kernel_name": "env_kernel", "instance_number": 1}], "env_trace")
        self._create_targets_file([{"kernel_name": "explicit_kernel", "instance_number": 2}], "explicit_trace")

        os.environ["NCOMPASS_TRACE_NAME"] = "env_trace"

        # Pass explicit trace_file_name
        result = load_ncu_kernel_targets(self.cache_dir, "explicit_trace")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["kernel_name"], "explicit_kernel")

    def test_no_env_var_no_trace_name_returns_all(self):
        """Test that without env var or trace_file_name, returns all targets."""
        self._create_targets_file([{"kernel_name": "kernel_a", "instance_number": 1}], "trace_a")
        self._create_targets_file([{"kernel_name": "kernel_b", "instance_number": 2}], "trace_b")

        # No NCOMPASS_TRACE_NAME set, pass None
        result = load_ncu_kernel_targets(self.cache_dir, None)

        self.assertEqual(len(result), 2)
        kernel_names = [t["kernel_name"] for t in result]
        self.assertIn("kernel_a", kernel_names)
        self.assertIn("kernel_b", kernel_names)


if __name__ == "__main__":
    unittest.main()
