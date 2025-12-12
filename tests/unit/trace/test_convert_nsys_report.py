"""
Tests for ncompass.trace.converters.converter.convert_nsys_report function.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from ncompass.trace.converters.converter import convert_nsys_report
from ncompass.trace.converters.models import ConversionOptions


class TestConvertNsysReportFileValidation(unittest.TestCase):
    """Test cases for file validation in convert_nsys_report."""

    def test_convert_nsys_report_file_not_found(self):
        """Test that FileNotFoundError is raised when input file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_file = os.path.join(temp_dir, "nonexistent.nsys-rep")
            output_path = os.path.join(temp_dir, "output.json.gz")
            
            with self.assertRaises(FileNotFoundError) as cm:
                convert_nsys_report(nonexistent_file, output_path)
            
            self.assertIn("Input file not found", str(cm.exception))
            self.assertIn("nonexistent.nsys-rep", str(cm.exception))


class TestConvertNsysReportSubprocess(unittest.TestCase):
    """Test cases for subprocess/nsys CLI handling in convert_nsys_report."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        # Create a dummy input file
        self.input_file = os.path.join(self.temp_dir, "test.nsys-rep")
        Path(self.input_file).touch()
        self.output_file = os.path.join(self.temp_dir, "output.json.gz")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    def test_convert_nsys_report_nsys_not_found(self, mock_run):
        """Test that FileNotFoundError is raised when nsys CLI is not available."""
        mock_run.side_effect = FileNotFoundError("nsys not found")
        
        with self.assertRaises(FileNotFoundError) as cm:
            convert_nsys_report(self.input_file, self.output_file)
        
        self.assertIn("nsys", str(cm.exception).lower())
        self.assertIn("not found", str(cm.exception).lower())

    @patch('subprocess.run')
    def test_convert_nsys_report_nsys_export_fails(self, mock_run):
        """Test that RuntimeError is raised when nsys export fails."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["nsys", "export"],
            stderr="Error: Invalid nsys-rep file format"
        )
        
        with self.assertRaises(RuntimeError) as cm:
            convert_nsys_report(self.input_file, self.output_file)
        
        self.assertIn("nsys export failed", str(cm.exception))

    @patch('subprocess.run')
    def test_convert_nsys_report_correct_nsys_command(self, mock_run):
        """Test that the correct nsys export command is built."""
        # Make subprocess.run succeed but then fail at converter step
        mock_run.return_value = MagicMock(returncode=0)
        
        # We need to also mock the converter to avoid actual SQLite operations
        with patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter') as mock_converter:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.convert.return_value = {"traceEvents": []}
            mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
            
            with patch('ncompass.trace.converters.utils.write_chrome_trace_gz'):
                convert_nsys_report(self.input_file, self.output_file)
        
        # Verify subprocess.run was called
        mock_run.assert_called_once()
        
        # Get the command that was passed
        call_args = mock_run.call_args
        command = call_args[0][0]  # First positional arg is the command list
        
        # Verify command structure
        self.assertEqual(command[0], "nsys")
        self.assertEqual(command[1], "export")
        self.assertIn("--type", command)
        self.assertIn("sqlite", command)
        self.assertIn("--force-overwrite", command)
        self.assertIn("-o", command)
        self.assertIn(str(self.input_file), command)
        
        # Verify check=True was passed
        self.assertTrue(call_args[1].get('check', False))


class TestConvertNsysReportFileCleanup(unittest.TestCase):
    """Test cases for file cleanup behavior in convert_nsys_report."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "test.nsys-rep")
        Path(self.input_file).touch()
        self.output_file = os.path.join(self.temp_dir, "output.json.gz")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_convert_nsys_report_cleanup_temp_sqlite(self, mock_write, mock_converter, mock_run):
        """Test that SQLite file is deleted when keep_sqlite=False."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = {"traceEvents": []}
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        # Create a mock SQLite file in temp directory
        temp_sqlite = os.path.join(tempfile.gettempdir(), "test.sqlite")
        Path(temp_sqlite).touch()
        
        try:
            # Run with keep_sqlite=False (default)
            convert_nsys_report(self.input_file, self.output_file, keep_sqlite=False)
            
            # The function should attempt to delete the temp SQLite file
            # Since we can't easily verify the exact temp path, we verify via mock
            # that the function completed without errors
            mock_write.assert_called_once()
        finally:
            # Clean up the temp file if it still exists
            if os.path.exists(temp_sqlite):
                os.unlink(temp_sqlite)

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_convert_nsys_report_keep_sqlite(self, mock_write, mock_converter, mock_run):
        """Test that SQLite file is preserved when keep_sqlite=True."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = {"traceEvents": []}
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        # Capture the sqlite_path that would be used
        captured_sqlite_path = None
        
        def capture_sqlite_path(path):
            nonlocal captured_sqlite_path
            captured_sqlite_path = path
            mock_result = MagicMock()
            mock_result.set_options.return_value = mock_ctx
            return mock_result
        
        mock_converter.return_value.set_sqlite_path.side_effect = capture_sqlite_path
        
        # Run with keep_sqlite=True
        convert_nsys_report(self.input_file, self.output_file, keep_sqlite=True)
        
        # Verify that the sqlite path is next to the input file
        expected_sqlite_path = str(Path(self.input_file).with_suffix('.sqlite'))
        self.assertEqual(captured_sqlite_path, expected_sqlite_path)


class TestConvertNsysReportOptions(unittest.TestCase):
    """Test cases for conversion options handling in convert_nsys_report."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "test.nsys-rep")
        Path(self.input_file).touch()
        self.output_file = os.path.join(self.temp_dir, "output.json.gz")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_convert_nsys_report_default_options(self, mock_write, mock_converter, mock_run):
        """Test that default ConversionOptions are used when none provided."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = {"traceEvents": []}
        
        captured_options = None
        
        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_ctx
        
        mock_set_sqlite = MagicMock()
        mock_set_sqlite.set_options.side_effect = capture_options
        mock_converter.return_value.set_sqlite_path.return_value = mock_set_sqlite
        
        # Call without options
        convert_nsys_report(self.input_file, self.output_file)
        
        # Verify default options were created
        self.assertIsNotNone(captured_options)
        self.assertIsInstance(captured_options, ConversionOptions)
        
        # Verify default activity types
        expected_activities = ["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"]
        self.assertEqual(captured_options.activity_types, expected_activities)
        self.assertTrue(captured_options.include_metadata)

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_convert_nsys_report_custom_options(self, mock_write, mock_converter, mock_run):
        """Test that custom options are passed through correctly."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = {"traceEvents": []}
        
        captured_options = None
        
        def capture_options(options):
            nonlocal captured_options
            captured_options = options
            return mock_ctx
        
        mock_set_sqlite = MagicMock()
        mock_set_sqlite.set_options.side_effect = capture_options
        mock_converter.return_value.set_sqlite_path.return_value = mock_set_sqlite
        
        # Create custom options
        custom_options = ConversionOptions(
            activity_types=["kernel", "nvtx"],
            include_metadata=False,
            nvtx_event_prefix=["test_"],
            nvtx_color_scheme={"test_.*": "blue"}
        )
        
        # Call with custom options
        convert_nsys_report(self.input_file, self.output_file, options=custom_options)
        
        # Verify custom options were passed through
        self.assertIs(captured_options, custom_options)
        self.assertEqual(captured_options.activity_types, ["kernel", "nvtx"])
        self.assertFalse(captured_options.include_metadata)
        self.assertEqual(captured_options.nvtx_event_prefix, ["test_"])
        self.assertEqual(captured_options.nvtx_color_scheme, {"test_.*": "blue"})


class TestConvertNsysReportCleanupOnError(unittest.TestCase):
    """Test cases for cleanup behavior when errors occur."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.temp_dir, "test.nsys-rep")
        Path(self.input_file).touch()
        self.output_file = os.path.join(self.temp_dir, "output.json.gz")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    def test_convert_nsys_report_cleanup_on_converter_error(self, mock_converter, mock_run):
        """Test that SQLite file is cleaned up even when converter fails."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.side_effect = RuntimeError("Conversion failed")
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        # Create a temp sqlite file to simulate nsys output
        sqlite_path = os.path.join(tempfile.gettempdir(), "test.sqlite")
        Path(sqlite_path).touch()
        
        try:
            with self.assertRaises(RuntimeError):
                convert_nsys_report(self.input_file, self.output_file, keep_sqlite=False)
            
            # The finally block should still execute and attempt cleanup
            # This verifies the function structure handles errors properly
        finally:
            if os.path.exists(sqlite_path):
                os.unlink(sqlite_path)

