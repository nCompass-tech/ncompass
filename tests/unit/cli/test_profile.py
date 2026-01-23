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
Tests for ncompass.cli.profile module.

Tests the profile subcommand for running nsys profiling.
"""

import argparse
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ncompass.cli.profile import add_profile_parser, run_profile_command


class TestAddProfileParser(unittest.TestCase):
    """Test cases for add_profile_parser function."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(dest="command")

    def test_add_profile_parser_adds_subcommand(self):
        """Test that add_profile_parser registers profile subcommand."""
        add_profile_parser(self.subparsers)
        
        # profile command now needs --nsys or --ncu
        args = self.parser.parse_args(["profile", "--nsys"])
        self.assertEqual(args.command, "profile")

    def test_add_profile_parser_sets_func(self):
        """Test that profile subcommand has func attribute."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys"])
        self.assertTrue(hasattr(args, "func"))
        self.assertEqual(args.func, run_profile_command)

    def test_add_profile_parser_no_positional_required(self):
        """Test that profile subcommand works without positional args (command comes after --)."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys"])
        self.assertEqual(args.command, "profile")

    def test_add_profile_parser_default_values(self):
        """Test default values for optional arguments."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys"])
        
        self.assertIsNone(args.output)
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.convert)
        self.assertFalse(args.verbose)
        self.assertFalse(args.quiet)

    def test_add_profile_parser_output_short_flag(self):
        """Test -o short flag for output."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys", "-o", "custom_name"])
        self.assertEqual(args.output, "custom_name")

    def test_add_profile_parser_output_dir_short_flag(self):
        """Test -d short flag for output-dir."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys", "-d", "/tmp/traces"])
        self.assertEqual(args.output_dir, "/tmp/traces")

    def test_add_profile_parser_convert_short_flag(self):
        """Test -c short flag for convert."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys", "-c"])
        self.assertTrue(args.convert)

    def test_add_profile_parser_verbose_short_flag(self):
        """Test -v short flag for verbose."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys", "-v"])
        self.assertTrue(args.verbose)

    def test_add_profile_parser_quiet_short_flag(self):
        """Test -q short flag for quiet."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args(["profile", "--nsys", "-q"])
        self.assertTrue(args.quiet)

    def test_add_profile_parser_requires_nsys_or_ncu(self):
        """Test that either --nsys or --ncu is required."""
        add_profile_parser(self.subparsers)
        
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["profile"])

    def test_add_profile_parser_mutually_exclusive_nsys_ncu(self):
        """Test that --nsys and --ncu are mutually exclusive."""
        add_profile_parser(self.subparsers)
        
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["profile", "--nsys", "--ncu"])

    def test_add_profile_parser_multiple_options(self):
        """Test multiple profile options can be combined."""
        add_profile_parser(self.subparsers)
        
        args = self.parser.parse_args([
            "profile", "--nsys", "-v", "--convert", "-o", "output"
        ])
        self.assertTrue(args.verbose)
        self.assertTrue(args.convert)
        self.assertEqual(args.output, "output")


class TestRunProfileCommandSuccess(unittest.TestCase):
    """Positive test cases for run_profile_command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_args(self, **kwargs):
        """Create argparse.Namespace with default values."""
        defaults = {
            "user_command": ["python", "test_script.py"],
            "extra_args": [],
            "nsys": True,
            "ncu": False,
            "output": None,
            "output_dir": None,
            "convert": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("ncompass.cli.profile.run_ncu_profile")
    @patch("ncompass.cli.profile.check_ncu_available")
    @patch("ncompass.cli.profile.convert_ncu_to_csv")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_ncu_success(
        self, mock_create_dir, mock_convert_ncu, mock_check_ncu, mock_run_ncu
    ):
        """Test successful NCU profiling returns 0."""
        mock_check_ncu.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        output_rep = Path(self.temp_dir) / "output.ncu-rep"
        output_rep.touch()
        mock_run_ncu.return_value = output_rep
        
        args = self._create_args(ncu=True, nsys=False, extra_args=["--nvtx-include", "range/"])
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 0)
        mock_run_ncu.assert_called_once()
        call_kwargs = mock_run_ncu.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--nvtx-include", "range/"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_success(self, mock_create_dir, mock_check_nsys, mock_run_nsys):
        """Test successful profiling returns 0."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args()
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 0)
        mock_run_nsys.assert_called_once()

    @patch("ncompass.cli.profile.convert_nsys_report")
    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_with_convert(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys, mock_convert
    ):
        """Test --convert flag triggers conversion after profiling."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(convert=True)
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 0)
        mock_convert.assert_called_once()

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_custom_output(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test custom output name is passed to run_nsys_profile."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "custom.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(output="custom")
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["output_name"], "custom")

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    def test_run_profile_command_custom_output_dir(self, mock_check_nsys, mock_run_nsys):
        """Test custom output directory is used."""
        mock_check_nsys.return_value = True
        output_dir = Path(self.temp_dir) / "custom_output"
        nsys_rep = output_dir / "output.nsys-rep"
        
        def side_effect(**kwargs):
            nsys_rep.touch()
            return nsys_rep
        
        mock_run_nsys.side_effect = side_effect
        
        args = self._create_args(output_dir=str(output_dir))
        
        run_profile_command(args)
        
        self.assertTrue(output_dir.exists())
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["trace_dir"], output_dir)

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_trace_types(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test custom trace types are passed correctly via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--trace=cuda,nvtx"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--trace=cuda,nvtx"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_with_range(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test range capture is enabled by default via defaults, or customized via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--capture-range=nvtx"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--capture-range=nvtx"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_no_python_tracing(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test python tracing can be disabled via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--python-sampling=false"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--python-sampling=false"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_cuda_graph_trace(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test --cuda-graph-trace mode is passed correctly via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--cuda-graph-trace=graph"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--cuda-graph-trace=graph"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_sudo(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test sudo usage can be controlled via extra_args (if implemented there)."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--sudo"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--sudo"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_user_command(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test user command is passed through correctly."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(user_command=["python", "train.py", "--epochs", "20", "--batch-size", "64"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["command"], ["python", "train.py", "--epochs", "20", "--batch-size", "64"])


class TestRunProfileCommandNegative(unittest.TestCase):
    """Negative test cases for run_profile_command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_args(self, **kwargs):
        """Create argparse.Namespace with default values."""
        defaults = {
            "user_command": ["python", "test.py"],
            "extra_args": [],
            "nsys": True,
            "ncu": False,
            "output": None,
            "output_dir": None,
            "convert": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_run_profile_command_no_command(self):
        """Test that missing command returns 1."""
        args = self._create_args(user_command=[])
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.profile.check_nsys_available")
    def test_run_profile_command_nsys_not_available(self, mock_check_nsys):
        """Test that missing nsys returns 1."""
        mock_check_nsys.return_value = False
        
        args = self._create_args()
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.profile.check_ncu_available")
    def test_run_profile_command_ncu_not_available(self, mock_check_ncu):
        """Test that missing ncu returns 1."""
        mock_check_ncu.return_value = False
        
        args = self._create_args(ncu=True, nsys=False)
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.profile.run_ncu_profile")
    @patch("ncompass.cli.profile.check_ncu_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_ncu_fails(
        self, mock_create_dir, mock_check_ncu, mock_run_ncu
    ):
        """Test that NCU profiling failure returns 1."""
        mock_check_ncu.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        mock_run_ncu.return_value = None  # Indicates failure
        
        args = self._create_args(ncu=True, nsys=False)
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_profiling_fails(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test that profiling failure returns 1."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        mock_run_nsys.return_value = None  # Indicates failure
        
        args = self._create_args()
        
        result = run_profile_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.profile.convert_nsys_report")
    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    @patch("ncompass.cli.profile.logger")
    def test_run_profile_command_conversion_fails(
        self, mock_logger, mock_create_dir, mock_check_nsys, mock_run_nsys, mock_convert
    ):
        """Test that conversion failure logs warning but returns 0."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        mock_convert.side_effect = Exception("Conversion failed")
        
        args = self._create_args(convert=True)
        
        result = run_profile_command(args)
        
        # Profiling succeeded, so should return 0 even if conversion fails
        self.assertEqual(result, 0)
        mock_logger.warning.assert_called()


class TestRunProfileCommandEdgeCases(unittest.TestCase):
    """Edge case tests for run_profile_command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_args(self, **kwargs):
        """Create argparse.Namespace with default values."""
        defaults = {
            "user_command": ["python", "test_script.py"],
            "extra_args": [],
            "nsys": True,
            "ncu": False,
            "output": None,
            "output_dir": None,
            "convert": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    @patch("ncompass.cli.profile.logger")
    def test_run_profile_command_verbose_logging(
        self, mock_logger, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test verbose flag sets DEBUG logging level."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(verbose=True)
        
        run_profile_command(args)
        
        mock_logger.setLevel.assert_called_with(logging.DEBUG)

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    @patch("ncompass.cli.profile.logger")
    def test_run_profile_command_quiet_logging(
        self, mock_logger, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test quiet flag sets ERROR logging level."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(quiet=True)
        
        run_profile_command(args)
        
        mock_logger.setLevel.assert_called_with(logging.ERROR)

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_no_force(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test --no-force flag can be passed via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--force-overwrite=false"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--force-overwrite=false"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_cache_dir(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test cache dir can be set (handled via environment or extra_args in real usage)."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        # In the new CLI, cache_dir would be an extra arg or environment variable
        args = self._create_args(extra_args=["--cache-dir=/tmp/cache"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--cache-dir=/tmp/cache"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_no_gpu_ctx_switch(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test --gpuctxsw flag can be passed via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--gpuctxsw=false"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--gpuctxsw=false"])

    @patch("ncompass.cli.profile.run_nsys_profile")
    @patch("ncompass.cli.profile.check_nsys_available")
    @patch("ncompass.cli.profile.create_trace_directory")
    def test_run_profile_command_no_cuda_memory_usage(
        self, mock_create_dir, mock_check_nsys, mock_run_nsys
    ):
        """Test --cuda-memory-usage flag can be passed via extra_args."""
        mock_check_nsys.return_value = True
        mock_create_dir.return_value = (Path(self.temp_dir), "20251205_120000")
        nsys_rep = Path(self.temp_dir) / "output.nsys-rep"
        nsys_rep.touch()
        mock_run_nsys.return_value = nsys_rep
        
        args = self._create_args(extra_args=["--cuda-memory-usage=false"])
        
        run_profile_command(args)
        
        call_kwargs = mock_run_nsys.call_args[1]
        self.assertEqual(call_kwargs["extra_args"], ["--cuda-memory-usage=false"])


if __name__ == "__main__":
    unittest.main()

