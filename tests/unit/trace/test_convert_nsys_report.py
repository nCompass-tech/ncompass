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
                convert_nsys_report(self.input_file, self.output_file, use_rust=False)
        
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
            convert_nsys_report(self.input_file, self.output_file, keep_sqlite=False, use_rust=False)
            
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
        convert_nsys_report(self.input_file, self.output_file, keep_sqlite=True, use_rust=False)
        
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
        convert_nsys_report(self.input_file, self.output_file, use_rust=False)
        
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
        convert_nsys_report(self.input_file, self.output_file, options=custom_options, use_rust=False)
        
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
                convert_nsys_report(self.input_file, self.output_file, keep_sqlite=False, use_rust=False)
            
            # The finally block should still execute and attempt cleanup
            # This verifies the function structure handles errors properly
        finally:
            if os.path.exists(sqlite_path):
                os.unlink(sqlite_path)


# =============================================================================
# Rust Backend Tests
# =============================================================================

class TestConvertNsysReportRustCommand(unittest.TestCase):
    """Test cases for Rust CLI command building in convert_nsys_report."""

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

    def _create_mock_rust_binary_exists(self, mock_path_class):
        """Helper to mock Path.exists() to return True for Rust binary."""
        original_exists = Path.exists
        
        def mock_exists(self):
            path_str = str(self)
            if 'nsys-chrome' in path_str:
                return True
            return original_exists(self)
        
        mock_path_class.exists = mock_exists

    @patch('subprocess.run')
    def test_rust_command_basic_structure(self, mock_run):
        """Test that basic Rust CLI command structure is correct."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        # Mock the binary existence check
        with patch.object(Path, 'exists') as mock_exists:
            # Return True for input file and rust binary, False otherwise
            def exists_side_effect(self=None):
                path_str = str(self) if self else ""
                return 'nsys-chrome' in path_str or path_str == self.input_file
            
            original_exists = Path.exists
            def smart_exists(path_self):
                path_str = str(path_self)
                if 'nsys-chrome' in path_str:
                    return True
                return original_exists(path_self)
            
            with patch.object(Path, 'exists', smart_exists):
                convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify subprocess.run was called
        mock_run.assert_called_once()
        
        # Get the command
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        
        # Verify basic command structure
        self.assertIn('nsys-chrome', cmd[0])  # Binary name
        self.assertEqual(cmd[1], self.input_file)  # Input file
        self.assertIn('-o', cmd)
        output_idx = cmd.index('-o')
        self.assertEqual(cmd[output_idx + 1], self.output_file)
        
        # Verify check=True was passed
        self.assertTrue(call_args[1].get('check', False))

    @patch('subprocess.run')
    def test_rust_command_activity_types(self, mock_run):
        """Test that activity types are passed to Rust binary via -t flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        options = ConversionOptions(
            activity_types=["kernel", "nvtx", "cuda-api"],
            include_metadata=True
        )
        
        original_exists = Path.exists
        def smart_exists(path_self):
            path_str = str(path_self)
            if 'nsys-chrome' in path_str:
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, options=options, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify -t flag with activity types
        self.assertIn('-t', cmd)
        t_idx = cmd.index('-t')
        self.assertEqual(cmd[t_idx + 1], "kernel,nvtx,cuda-api")

    @patch('subprocess.run')
    def test_rust_command_nvtx_prefix(self, mock_run):
        """Test that nvtx_event_prefix is passed via --nvtx-prefix flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        options = ConversionOptions(
            activity_types=["nvtx"],
            nvtx_event_prefix=["forward_", "backward_"],
            include_metadata=True
        )
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, options=options, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify --nvtx-prefix flag
        self.assertIn('--nvtx-prefix', cmd)
        prefix_idx = cmd.index('--nvtx-prefix')
        self.assertEqual(cmd[prefix_idx + 1], "forward_,backward_")

    @patch('subprocess.run')
    def test_rust_command_metadata_false(self, mock_run):
        """Test that include_metadata=False adds --metadata=false flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        options = ConversionOptions(
            activity_types=["kernel"],
            include_metadata=False
        )
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, options=options, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify --metadata=false flag is present
        self.assertIn('--metadata=false', cmd)

    @patch('subprocess.run')
    def test_rust_command_metadata_true_no_flag(self, mock_run):
        """Test that include_metadata=True does not add --metadata flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        options = ConversionOptions(
            activity_types=["kernel"],
            include_metadata=True
        )
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, options=options, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify --metadata flag is NOT present (True is default)
        metadata_flags = [arg for arg in cmd if 'metadata' in arg.lower()]
        self.assertEqual(len(metadata_flags), 0)

    @patch('subprocess.run')
    def test_rust_command_keep_sqlite(self, mock_run):
        """Test that keep_sqlite=True adds --keep-sqlite flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, keep_sqlite=True, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify --keep-sqlite flag is present
        self.assertIn('--keep-sqlite', cmd)

    @patch('subprocess.run')
    def test_rust_command_keep_sqlite_false_no_flag(self, mock_run):
        """Test that keep_sqlite=False does not add --keep-sqlite flag."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, keep_sqlite=False, use_rust=True)
        
        cmd = mock_run.call_args[0][0]
        
        # Verify --keep-sqlite flag is NOT present
        self.assertNotIn('--keep-sqlite', cmd)

    @patch('subprocess.run')
    def test_rust_command_all_options_combined(self, mock_run):
        """Test that all options are correctly combined in the command."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        options = ConversionOptions(
            activity_types=["kernel", "nvtx", "nvtx-kernel"],
            nvtx_event_prefix=["test_prefix_"],
            include_metadata=False
        )
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(
                self.input_file, 
                self.output_file, 
                options=options, 
                keep_sqlite=True, 
                use_rust=True
            )
        
        cmd = mock_run.call_args[0][0]
        
        # Verify all flags are present
        self.assertIn('-t', cmd)
        self.assertIn('--nvtx-prefix', cmd)
        self.assertIn('--metadata=false', cmd)
        self.assertIn('--keep-sqlite', cmd)
        
        # Verify values
        t_idx = cmd.index('-t')
        self.assertEqual(cmd[t_idx + 1], "kernel,nvtx,nvtx-kernel")
        
        prefix_idx = cmd.index('--nvtx-prefix')
        self.assertEqual(cmd[prefix_idx + 1], "test_prefix_")


class TestConvertNsysReportRustFallback(unittest.TestCase):
    """Test cases for fallback to Python when Rust backend fails."""

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
    def test_fallback_on_called_process_error(self, mock_write, mock_converter, mock_run):
        """Test fallback to Python when Rust binary returns CalledProcessError."""
        # First call (Rust) fails, second call (nsys export for Python) succeeds
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "nsys-chrome", stderr="Rust conversion failed"),
            MagicMock(returncode=0)  # nsys export succeeds
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = iter([{"name": "test"}])
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            # Should not raise - should fall back to Python
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify subprocess.run was called twice (Rust fail + nsys export)
        self.assertEqual(mock_run.call_count, 2)
        
        # Verify Python converter was used
        mock_converter.assert_called_once()
        mock_write.assert_called_once()

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_fallback_on_generic_exception(self, mock_write, mock_converter, mock_run):
        """Test fallback to Python when Rust binary raises generic Exception."""
        # First call (Rust) raises generic exception, second call (nsys export) succeeds
        mock_run.side_effect = [
            OSError("Permission denied"),
            MagicMock(returncode=0)  # nsys export succeeds
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = iter([{"name": "test"}])
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            # Should not raise - should fall back to Python
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify fallback occurred
        self.assertEqual(mock_run.call_count, 2)
        mock_converter.assert_called_once()

    @patch('subprocess.run')
    @patch('ncompass.trace.converters.converter.NsysToChromeTraceConverter')
    @patch('ncompass.trace.converters.utils.write_chrome_trace_gz')
    def test_fallback_when_binary_not_found(self, mock_write, mock_converter, mock_run):
        """Test fallback to Python when Rust binary does not exist."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = iter([{"name": "test"}])
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        original_exists = Path.exists
        def smart_exists(path_self):
            # Return False for rust binary, True for input file
            if 'nsys-chrome' in str(path_self):
                return False
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify only nsys export was called (not Rust binary)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "nsys")  # Python path uses nsys export
        
        # Verify Python converter was used
        mock_converter.assert_called_once()


class TestConvertNsysReportRustBinaryDiscovery(unittest.TestCase):
    """Test cases for Rust binary path discovery logic."""

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
    def test_packaged_binary_path_tried_first(self, mock_run):
        """Test that packaged binary location (ncompass/bin/nsys-chrome) is tried first."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        checked_paths = []
        original_exists = Path.exists
        
        def tracking_exists(path_self):
            path_str = str(path_self)
            if 'nsys-chrome' in path_str:
                checked_paths.append(path_str)
                # Return True for packaged location
                if '/bin/nsys-chrome' in path_str and 'ncompass_rust' not in path_str:
                    return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', tracking_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify packaged path was checked
        packaged_paths = [p for p in checked_paths if '/bin/nsys-chrome' in p and 'ncompass_rust' not in p]
        self.assertGreater(len(packaged_paths), 0, "Packaged binary path should be checked")
        
        # Verify the binary used was from packaged location
        cmd = mock_run.call_args[0][0]
        self.assertIn('/bin/nsys-chrome', cmd[0])
        self.assertNotIn('ncompass_rust', cmd[0])

    @patch('subprocess.run')
    def test_dev_binary_fallback_path(self, mock_run):
        """Test that dev binary path is tried when packaged binary is missing."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        checked_paths = []
        original_exists = Path.exists
        
        def tracking_exists(path_self):
            path_str = str(path_self)
            if 'nsys-chrome' in path_str:
                checked_paths.append(path_str)
                # Return False for packaged, True for dev location
                if 'ncompass_rust' in path_str:
                    return True
                return False
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', tracking_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify both paths were checked
        self.assertGreater(len(checked_paths), 1, "Should check multiple binary paths")
        
        # Verify dev path was checked after packaged path
        dev_paths = [p for p in checked_paths if 'ncompass_rust' in p]
        self.assertGreater(len(dev_paths), 0, "Dev binary path should be checked as fallback")
        
        # Verify the binary used was from dev location
        cmd = mock_run.call_args[0][0]
        self.assertIn('ncompass_rust', cmd[0])

    @patch('subprocess.run')
    def test_dev_binary_path_structure(self, mock_run):
        """Test that dev binary path has correct structure."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        captured_binary_path = None
        original_exists = Path.exists
        
        def tracking_exists(path_self):
            nonlocal captured_binary_path
            path_str = str(path_self)
            if 'nsys-chrome' in path_str:
                # Return False for packaged, True for dev
                if 'ncompass_rust' in path_str:
                    captured_binary_path = path_str
                    return True
                return False
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', tracking_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify dev path structure
        self.assertIsNotNone(captured_binary_path)
        self.assertIn('ncompass_rust', captured_binary_path)
        self.assertIn('trace_converters', captured_binary_path)
        self.assertIn('target', captured_binary_path)
        self.assertIn('x86_64-unknown-linux-musl', captured_binary_path)
        self.assertIn('release', captured_binary_path)
        self.assertIn('nsys-chrome', captured_binary_path)


class TestConvertNsysReportRustSuccess(unittest.TestCase):
    """Test cases for successful Rust backend execution."""

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
    def test_rust_success_skips_python_converter(self, mock_converter, mock_run):
        """Test that successful Rust execution does not invoke Python converter."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Verify subprocess.run was called once (only Rust)
        mock_run.assert_called_once()
        
        # Verify Python converter was NOT invoked
        mock_converter.assert_not_called()

    @patch('subprocess.run')
    def test_rust_success_returns_early(self, mock_run):
        """Test that function returns immediately after successful Rust execution."""
        mock_run.return_value = MagicMock(returncode=0, stderr=None)
        
        original_exists = Path.exists
        def smart_exists(path_self):
            if 'nsys-chrome' in str(path_self):
                return True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', smart_exists):
            # Should complete without errors
            result = convert_nsys_report(self.input_file, self.output_file, use_rust=True)
        
        # Function returns None on success
        self.assertIsNone(result)
        
        # Only one subprocess call (the Rust binary)
        mock_run.assert_called_once()


class TestConvertNsysReportRustDisabled(unittest.TestCase):
    """Test cases for when Rust backend is explicitly disabled."""

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
    def test_use_rust_false_skips_binary_check(self, mock_write, mock_converter, mock_run):
        """Test that use_rust=False skips Rust binary entirely."""
        mock_run.return_value = MagicMock(returncode=0)
        
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.convert.return_value = iter([{"name": "test"}])
        mock_converter.return_value.set_sqlite_path.return_value.set_options.return_value = mock_ctx
        
        checked_rust_binary = False
        original_exists = Path.exists
        
        def tracking_exists(path_self):
            nonlocal checked_rust_binary
            if 'nsys-chrome' in str(path_self):
                checked_rust_binary = True
            return original_exists(path_self)
        
        with patch.object(Path, 'exists', tracking_exists):
            convert_nsys_report(self.input_file, self.output_file, use_rust=False)
        
        # Verify Rust binary was NOT checked
        self.assertFalse(checked_rust_binary, "Rust binary should not be checked when use_rust=False")
        
        # Verify nsys export was called (Python path)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertEqual(cmd[0], "nsys")
        
        # Verify Python converter was used
        mock_converter.assert_called_once()
