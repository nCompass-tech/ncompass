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
Tests for ncompass.cli.convert module.

Tests the convert subcommand for nsys-rep to Chrome trace conversion.
"""

import argparse
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ncompass.cli.convert import add_convert_parser, run_convert_command
from ncompass.trace.converters.models import ConversionOptions


class TestAddConvertParser(unittest.TestCase):
    """Test cases for add_convert_parser function."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(dest="command")

    def test_add_convert_parser_adds_subcommand(self):
        """Test that add_convert_parser registers convert subcommand."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep"])
        self.assertEqual(args.command, "convert")

    def test_add_convert_parser_sets_func(self):
        """Test that convert subcommand has func attribute."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep"])
        self.assertTrue(hasattr(args, "func"))
        self.assertEqual(args.func, run_convert_command)

    def test_add_convert_parser_input_file_required(self):
        """Test that input_file argument is required."""
        add_convert_parser(self.subparsers)
        
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["convert"])

    def test_add_convert_parser_default_values(self):
        """Test default values for optional arguments."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep"])
        
        self.assertEqual(args.input_file, "test.nsys-rep")
        self.assertIsNone(args.output)
        self.assertIsNone(args.output_dir)
        self.assertFalse(args.keep_sqlite)
        self.assertEqual(args.activity_types, "kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched")
        self.assertFalse(args.no_metadata)
        self.assertFalse(args.verbose)
        self.assertFalse(args.quiet)

    def test_add_convert_parser_output_short_flag(self):
        """Test -o short flag for output."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep", "-o", "custom_name"])
        self.assertEqual(args.output, "custom_name")

    def test_add_convert_parser_output_dir_short_flag(self):
        """Test -d short flag for output-dir."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep", "-d", "/tmp/traces"])
        self.assertEqual(args.output_dir, "/tmp/traces")

    def test_add_convert_parser_activity_types_short_flag(self):
        """Test -a short flag for activity-types."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep", "-a", "kernel,nvtx"])
        self.assertEqual(args.activity_types, "kernel,nvtx")

    def test_add_convert_parser_verbose_short_flag(self):
        """Test -v short flag for verbose."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep", "-v"])
        self.assertTrue(args.verbose)

    def test_add_convert_parser_quiet_short_flag(self):
        """Test -q short flag for quiet."""
        add_convert_parser(self.subparsers)
        
        args = self.parser.parse_args(["convert", "test.nsys-rep", "-q"])
        self.assertTrue(args.quiet)


class TestRunConvertCommandSuccess(unittest.TestCase):
    """Positive test cases for run_convert_command."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / "test.nsys-rep"
        self.input_file.touch()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_args(self, **kwargs):
        """Create argparse.Namespace with default values."""
        defaults = {
            "input_file": str(self.input_file),
            "output": None,
            "output_dir": None,
            "keep_sqlite": False,
            "activity_types": "kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched",
            "no_metadata": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_success(self, mock_convert):
        """Test successful conversion returns 0."""
        args = self._create_args()
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        mock_convert.assert_called_once()

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_custom_output(self, mock_convert):
        """Test custom output name is used."""
        args = self._create_args(output="custom_name")
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        call_kwargs = mock_convert.call_args[1]
        self.assertIn("custom_name.json.gz", call_kwargs["output_path"])

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_custom_output_dir(self, mock_convert):
        """Test custom output directory is used."""
        output_dir = Path(self.temp_dir) / "custom_output"
        args = self._create_args(output_dir=str(output_dir))
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        # Verify directory was created
        self.assertTrue(output_dir.exists())
        # Verify output path uses custom directory
        call_kwargs = mock_convert.call_args[1]
        self.assertTrue(call_kwargs["output_path"].startswith(str(output_dir)))

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_keep_sqlite(self, mock_convert):
        """Test keep_sqlite flag is passed correctly."""
        args = self._create_args(keep_sqlite=True)
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        call_kwargs = mock_convert.call_args[1]
        self.assertTrue(call_kwargs["keep_sqlite"])

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_activity_types(self, mock_convert):
        """Test custom activity types are parsed correctly."""
        args = self._create_args(activity_types="kernel,nvtx")
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        call_kwargs = mock_convert.call_args[1]
        options = call_kwargs["options"]
        self.assertEqual(options.activity_types, ["kernel", "nvtx"])

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_no_metadata(self, mock_convert):
        """Test no_metadata flag affects options."""
        args = self._create_args(no_metadata=True)
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 0)
        call_kwargs = mock_convert.call_args[1]
        options = call_kwargs["options"]
        self.assertFalse(options.include_metadata)

    @patch("ncompass.cli.convert.convert_nsys_report")
    @patch("ncompass.cli.convert.logger")
    def test_run_convert_command_verbose(self, mock_logger, mock_convert):
        """Test verbose flag sets DEBUG logging level."""
        args = self._create_args(verbose=True)
        
        run_convert_command(args)
        
        mock_logger.setLevel.assert_called_with(logging.DEBUG)

    @patch("ncompass.cli.convert.convert_nsys_report")
    @patch("ncompass.cli.convert.logger")
    def test_run_convert_command_quiet(self, mock_logger, mock_convert):
        """Test quiet flag sets ERROR logging level."""
        args = self._create_args(quiet=True)
        
        run_convert_command(args)
        
        mock_logger.setLevel.assert_called_with(logging.ERROR)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_default_output_dir(self, mock_convert):
        """Test default output directory is input file's parent."""
        args = self._create_args()
        
        run_convert_command(args)
        
        call_kwargs = mock_convert.call_args[1]
        expected_dir = str(self.input_file.parent)
        self.assertTrue(call_kwargs["output_path"].startswith(expected_dir))


class TestRunConvertCommandNegative(unittest.TestCase):
    """Negative test cases for run_convert_command."""

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
            "input_file": str(Path(self.temp_dir) / "nonexistent.nsys-rep"),
            "output": None,
            "output_dir": None,
            "keep_sqlite": False,
            "activity_types": "kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched",
            "no_metadata": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_run_convert_command_file_not_found(self):
        """Test that missing input file returns 1."""
        args = self._create_args()
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_conversion_failure_file_not_found(self, mock_convert):
        """Test that FileNotFoundError during conversion returns 1."""
        # Create input file
        input_file = Path(self.temp_dir) / "test.nsys-rep"
        input_file.touch()
        
        mock_convert.side_effect = FileNotFoundError("nsys not found")
        args = self._create_args(input_file=str(input_file))
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_conversion_failure_runtime(self, mock_convert):
        """Test that RuntimeError during conversion returns 1."""
        input_file = Path(self.temp_dir) / "test.nsys-rep"
        input_file.touch()
        
        mock_convert.side_effect = RuntimeError("Conversion failed")
        args = self._create_args(input_file=str(input_file))
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 1)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_unexpected_error(self, mock_convert):
        """Test that unexpected Exception returns 1."""
        input_file = Path(self.temp_dir) / "test.nsys-rep"
        input_file.touch()
        
        mock_convert.side_effect = Exception("Unexpected error")
        args = self._create_args(input_file=str(input_file))
        
        result = run_convert_command(args)
        
        self.assertEqual(result, 1)


class TestRunConvertCommandEdgeCases(unittest.TestCase):
    """Edge case tests for run_convert_command."""

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
            "input_file": str(Path(self.temp_dir) / "test.nsys-rep"),
            "output": None,
            "output_dir": None,
            "keep_sqlite": False,
            "activity_types": "kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched",
            "no_metadata": False,
            "verbose": False,
            "quiet": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @patch("ncompass.cli.convert.convert_nsys_report")
    @patch("ncompass.cli.convert.logger")
    def test_run_convert_command_non_nsys_rep_extension(self, mock_logger, mock_convert):
        """Test that non-.nsys-rep extension logs a warning."""
        input_file = Path(self.temp_dir) / "test.txt"
        input_file.touch()
        
        args = self._create_args(input_file=str(input_file))
        
        run_convert_command(args)
        
        # Verify warning was logged
        mock_logger.warning.assert_called()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("should be a .nsys-rep file", warning_msg)

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_creates_output_dir(self, mock_convert):
        """Test that output directory is created if it doesn't exist."""
        input_file = Path(self.temp_dir) / "test.nsys-rep"
        input_file.touch()
        
        nested_output_dir = Path(self.temp_dir) / "nested" / "output" / "dir"
        args = self._create_args(
            input_file=str(input_file),
            output_dir=str(nested_output_dir)
        )
        
        # Directory should not exist yet
        self.assertFalse(nested_output_dir.exists())
        
        run_convert_command(args)
        
        # Directory should now exist
        self.assertTrue(nested_output_dir.exists())

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_activity_types_with_spaces(self, mock_convert):
        """Test that activity types with spaces are trimmed."""
        input_file = Path(self.temp_dir) / "test.nsys-rep"
        input_file.touch()
        
        args = self._create_args(
            input_file=str(input_file),
            activity_types="kernel , nvtx , cuda-api"
        )
        
        run_convert_command(args)
        
        call_kwargs = mock_convert.call_args[1]
        options = call_kwargs["options"]
        self.assertEqual(options.activity_types, ["kernel", "nvtx", "cuda-api"])

    @patch("ncompass.cli.convert.convert_nsys_report")
    def test_run_convert_command_uses_stem_for_default_output(self, mock_convert):
        """Test that default output name is derived from input file stem."""
        input_file = Path(self.temp_dir) / "my_trace.nsys-rep"
        input_file.touch()
        
        args = self._create_args(input_file=str(input_file))
        
        run_convert_command(args)
        
        call_kwargs = mock_convert.call_args[1]
        self.assertIn("my_trace.json.gz", call_kwargs["output_path"])


if __name__ == "__main__":
    unittest.main()

