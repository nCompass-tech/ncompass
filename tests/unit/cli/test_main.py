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
Tests for ncompass.cli.main module.

Tests the main CLI entry point including parser creation and command dispatch.
"""

import argparse
import sys
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch

from ncompass.cli.main import create_parser, main


class TestCreateParser(unittest.TestCase):
    """Test cases for create_parser function."""

    def test_create_parser_returns_parser(self):
        """Test that create_parser returns an ArgumentParser instance."""
        parser = create_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_create_parser_has_version_argument(self):
        """Test that parser has --version argument."""
        parser = create_parser()
        # Version argument will cause SystemExit when called
        with self.assertRaises(SystemExit):
            parser.parse_args(["--version"])

    def test_create_parser_has_subparsers(self):
        """Test that parser has profile and convert subcommands."""
        parser = create_parser()
        
        # Parse profile command (no positional args, user command comes after --)
        args = parser.parse_args(["profile"])
        self.assertEqual(args.command, "profile")
        
        # Parse convert command (needs required 'input_file' arg)
        args = parser.parse_args(["convert", "test.nsys-rep"])
        self.assertEqual(args.command, "convert")

    def test_create_parser_profile_has_func(self):
        """Test that profile subcommand sets func attribute."""
        parser = create_parser()
        args = parser.parse_args(["profile"])
        self.assertTrue(hasattr(args, "func"))

    def test_create_parser_convert_has_func(self):
        """Test that convert subcommand sets func attribute."""
        parser = create_parser()
        args = parser.parse_args(["convert", "test.nsys-rep"])
        self.assertTrue(hasattr(args, "func"))


class TestMainNoArgs(unittest.TestCase):
    """Test cases for main function with no arguments."""

    def test_main_no_args_returns_zero(self):
        """Test that main with no args returns 0."""
        result = main([])
        self.assertEqual(result, 0)

    @patch("sys.stdout", new_callable=StringIO)
    def test_main_no_args_prints_help(self, mock_stdout):
        """Test that main with no args prints help message."""
        # Use a fresh parser to capture help output
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = argparse.Namespace(command=None)
            mock_create_parser.return_value = mock_parser
            
            main([])
            
            mock_parser.print_help.assert_called_once()


class TestMainVersionFlag(unittest.TestCase):
    """Test cases for version flag handling."""

    def test_main_version_flag_short(self):
        """Test that -V flag triggers version display and exits."""
        with self.assertRaises(SystemExit) as cm:
            main(["-V"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_version_flag_long(self):
        """Test that --version flag triggers version display and exits."""
        with self.assertRaises(SystemExit) as cm:
            main(["--version"])
        self.assertEqual(cm.exception.code, 0)


class TestMainCommandDispatch(unittest.TestCase):
    """Test cases for command dispatch in main function."""

    @patch("ncompass.cli.profile.run_profile_command")
    def test_main_with_profile_command(self, mock_run_profile):
        """Test that profile command dispatches to run_profile_command."""
        mock_run_profile.return_value = 0
        
        # Need to patch to avoid actual file check
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_args = argparse.Namespace(command="profile", func=mock_run_profile)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            result = main(["profile", "--", "python", "test.py"])
            
            # Verify user_command was attached
            self.assertEqual(mock_args.user_command, ["python", "test.py"])
            mock_run_profile.assert_called_once_with(mock_args)
            self.assertEqual(result, 0)

    @patch("ncompass.cli.convert.run_convert_command")
    def test_main_with_convert_command(self, mock_run_convert):
        """Test that convert command dispatches to run_convert_command."""
        mock_run_convert.return_value = 0
        
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_args = argparse.Namespace(command="convert", func=mock_run_convert)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            result = main(["convert", "test.nsys-rep"])
            
            mock_run_convert.assert_called_once_with(mock_args)
            self.assertEqual(result, 0)


class TestMainSeparatorHandling(unittest.TestCase):
    """Test cases for -- separator handling."""

    @patch("ncompass.cli.profile.run_profile_command")
    def test_main_separator_splits_args(self, mock_run_profile):
        """Test that -- separator correctly splits ncompass args from user command."""
        mock_run_profile.return_value = 0
        
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_args = argparse.Namespace(command="profile", func=mock_run_profile)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            main(["profile", "--convert", "--", "python", "train.py", "--epochs", "10"])
            
            # Parser should only receive ncompass args (before --)
            mock_parser.parse_args.assert_called_once_with(["profile", "--convert"])
            # User command should be attached to args
            self.assertEqual(mock_args.user_command, ["python", "train.py", "--epochs", "10"])

    @patch("ncompass.cli.profile.run_profile_command")
    def test_main_no_separator_empty_user_command(self, mock_run_profile):
        """Test that without --, user_command is empty list."""
        mock_run_profile.return_value = 0
        
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_args = argparse.Namespace(command="profile", func=mock_run_profile)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            main(["profile", "--convert"])
            
            self.assertEqual(mock_args.user_command, [])

    @patch("ncompass.cli.profile.run_profile_command")
    def test_main_separator_with_flags_after(self, mock_run_profile):
        """Test that flags after -- are passed to user command, not parsed."""
        mock_run_profile.return_value = 0
        
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_args = argparse.Namespace(command="profile", func=mock_run_profile)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            # --verbose after -- should go to user command, not be parsed as ncompass flag
            main(["profile", "--", "./app", "--verbose", "-c"])
            
            mock_parser.parse_args.assert_called_once_with(["profile"])
            self.assertEqual(mock_args.user_command, ["./app", "--verbose", "-c"])


class TestMainNegativeCases(unittest.TestCase):
    """Negative test cases for main function."""

    def test_main_invalid_command(self):
        """Test that unknown command causes error exit."""
        with self.assertRaises(SystemExit) as cm:
            main(["unknown_command"])
        self.assertNotEqual(cm.exception.code, 0)

    def test_main_missing_func_attribute(self):
        """Test that missing func attribute returns 1."""
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            # Args with command but no func attribute
            mock_args = argparse.Namespace(command="profile")
            # Explicitly remove func if it exists
            if hasattr(mock_args, "func"):
                delattr(mock_args, "func")
            
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            result = main(["profile", "--", "python", "test.py"])
            
            mock_parser.print_help.assert_called_once()
            self.assertEqual(result, 1)


class TestMainHelpFlag(unittest.TestCase):
    """Test cases for help flag handling."""

    def test_main_help_flag_short(self):
        """Test that -h flag triggers help and exits."""
        with self.assertRaises(SystemExit) as cm:
            main(["-h"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_help_flag_long(self):
        """Test that --help flag triggers help and exits."""
        with self.assertRaises(SystemExit) as cm:
            main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_profile_help(self):
        """Test that profile --help exits with 0."""
        with self.assertRaises(SystemExit) as cm:
            main(["profile", "--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_main_convert_help(self):
        """Test that convert --help exits with 0."""
        with self.assertRaises(SystemExit) as cm:
            main(["convert", "--help"])
        self.assertEqual(cm.exception.code, 0)


class TestMainCommandHandlerReturn(unittest.TestCase):
    """Test cases for command handler return value propagation."""

    def test_main_propagates_handler_return_value(self):
        """Test that main returns the value from the command handler."""
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_handler = MagicMock(return_value=42)
            mock_args = argparse.Namespace(command="test", func=mock_handler)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            result = main(["test"])
            
            self.assertEqual(result, 42)

    def test_main_propagates_nonzero_exit_code(self):
        """Test that main returns non-zero exit codes from handlers."""
        with patch("ncompass.cli.main.create_parser") as mock_create_parser:
            mock_handler = MagicMock(return_value=1)
            mock_args = argparse.Namespace(command="test", func=mock_handler)
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser
            
            result = main(["test"])
            
            self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()

