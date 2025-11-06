"""
Tests for ncompass.trace.core.session module.
"""

import unittest
import tempfile
import os
import gzip
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from ncompass.trace.core.session import ProfilingSession


class TestProfilingSessionInit(unittest.TestCase):
    """Test cases for ProfilingSession initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        session = ProfilingSession(trace_output_dir=self.temp_dir)
        
        self.assertEqual(session.trace_output_dir, Path(self.temp_dir))
        self.assertIsInstance(session.cache_dir, str)
        self.assertEqual(session.session_name, "profiling_session")
        self.assertEqual(session.base_url, "http://localhost:8000")
        self.assertIsNone(session.latest_trace_path)
        self.assertIsNone(session.latest_trace_name)
        self.assertIsNone(session.latest_trace_summary)
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        cache_dir = os.path.join(self.temp_dir, 'cache')
        session = ProfilingSession(
            trace_output_dir=self.temp_dir,
            cache_dir=cache_dir,
            session_name="test_session",
            base_url="http://example.com:9000"
        )
        
        self.assertEqual(session.cache_dir, cache_dir)
        self.assertEqual(session.session_name, "test_session")
        self.assertEqual(session.base_url, "http://example.com:9000")
    
    def test_init_creates_config_manager(self):
        """Test that initialization creates ConfigManager."""
        session = ProfilingSession(trace_output_dir=self.temp_dir)
        
        self.assertIsNotNone(session.config_manager)
        self.assertIsInstance(session.config_manager.cache_dir, str)


class TestProfilingSessionRunProfile(unittest.TestCase):
    """Test cases for run_profile method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(trace_output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('ncompass.trace.core.session.ProfilingSession._find_and_rename_latest_trace')
    def test_run_profile_success(self, mock_find_rename):
        """Test successful profile run."""
        mock_find_rename.return_value = f"{self.temp_dir}/test_trace.pt.trace.json"
        
        def user_code():
            pass
        
        result = self.session.run_profile(user_code)
        
        self.assertIsNotNone(self.session.latest_trace_name)
        self.assertIn("profiling_session_iter0", self.session.latest_trace_name)
        self.assertEqual(result, mock_find_rename.return_value)
        mock_find_rename.assert_called_once()
    
    @patch('ncompass.trace.core.session.ProfilingSession._find_and_rename_latest_trace')
    def test_run_profile_with_suffix(self, mock_find_rename):
        """Test profile run with trace name suffix."""
        mock_find_rename.return_value = f"{self.temp_dir}/test_trace.pt.trace.json"
        
        def user_code():
            pass
        
        result = self.session.run_profile(user_code, trace_name_suffix="custom")
        
        self.assertIn("custom", self.session.latest_trace_name)
        self.assertIn("profiling_session", self.session.latest_trace_name)
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    @patch('ncompass.trace.core.session.ProfilingSession._find_and_rename_latest_trace')
    def test_run_profile_with_filter(self, mock_find_rename, mock_submit):
        """Test profile run with trace filtering."""
        initial_trace = f"{self.temp_dir}/test_trace.pt.trace.json"
        filtered_trace = f"{self.temp_dir}/test_trace_filtered.pt.trace.json"
        
        mock_find_rename.return_value = initial_trace
        mock_submit.return_value = {'filtered_trace_path': filtered_trace}
        
        def user_code():
            pass
        
        result = self.session.run_profile(
            user_code,
            filter_trace=True,
            filter_trace_args={'min_duration': 100}
        )
        
        self.assertEqual(result, filtered_trace)
        mock_submit.assert_called_once()
        call_args = mock_submit.call_args[1]
        self.assertEqual(call_args['endpoint'], 'filter')
        self.assertTrue(call_args['await_result'])
    
    @patch('ncompass.trace.core.session.ProfilingSession._find_and_rename_latest_trace')
    def test_run_profile_user_code_exception(self, mock_find_rename):
        """Test that exceptions in user code are propagated."""
        def failing_code():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError) as cm:
            self.session.run_profile(failing_code)
        
        self.assertIn("Test error", str(cm.exception))
        # Should not call find_and_rename since exception occurred
        mock_find_rename.assert_not_called()


class TestProfilingSessionTraceSummary(unittest.TestCase):
    """Test cases for trace summary methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(
            trace_output_dir=self.temp_dir,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    def test_get_trace_summary_success(self, mock_submit):
        """Test successful trace summary generation."""
        trace_path = f"{self.temp_dir}/test.pt.trace.json"
        self.session.latest_trace_path = trace_path
        
        mock_summary = {
            'markdown': '# Summary\n\nTest summary',
            'structured': {'key': 'value'}
        }
        mock_submit.return_value = mock_summary
        
        with patch.object(self.session, 'save_trace_summary'):
            result = self.session.get_trace_summary()
        
        self.assertEqual(result, mock_summary)
        self.assertEqual(self.session.latest_trace_summary, mock_summary)
        mock_submit.assert_called_once()
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    def test_get_trace_summary_with_feedback_context(self, mock_submit):
        """Test trace summary with feedback context."""
        trace_path = f"{self.temp_dir}/test.pt.trace.json"
        feedback_context = {
            'feedback_text': 'Why is this slow?',
            'target_module': 'test.module',
            'start_line': 10,
            'end_line': 20
        }
        
        mock_submit.return_value = {'markdown': 'Summary'}
        
        with patch.object(self.session, 'save_trace_summary'):
            self.session.get_trace_summary(trace_path, feedback_context)
        
        call_kwargs = mock_submit.call_args[1]  # or use .kwargs
        self.assertEqual(call_kwargs['request']['feedback_text'], 'Why is this slow?')
        self.assertEqual(call_kwargs['request']['target_module'], 'test.module')
    
    def test_get_trace_summary_no_trace(self):
        """Test get_trace_summary raises error when no trace available."""
        with self.assertRaises(ValueError) as cm:
            self.session.get_trace_summary()
        
        self.assertIn("No trace file available", str(cm.exception))
    
    @patch('ncompass.trace.core.session.ConfigManager.save_trace_summary')
    def test_save_trace_summary(self, mock_save):
        """Test save_trace_summary method."""
        summary = {'markdown': 'Test', 'data': {}}
        trace_path = f"{self.temp_dir}/test.pt.trace.json"
        trace_name = "test_trace"
        
        self.session.latest_trace_summary = summary
        self.session.latest_trace_path = trace_path
        self.session.latest_trace_name = trace_name
        
        mock_save.return_value = ('/path/to/json', '/path/to/md')
        
        json_path, md_path = self.session.save_trace_summary()
        
        mock_save.assert_called_once_with(
            summary=summary,
            trace_path=trace_path,
            trace_name=trace_name
        )
    
    def test_save_trace_summary_no_summary(self):
        """Test save_trace_summary raises error when no summary."""
        with self.assertRaises(ValueError) as cm:
            self.session.save_trace_summary()
        
        self.assertIn("No trace summary available", str(cm.exception))


class TestProfilingSessionFeedback(unittest.TestCase):
    """Test cases for feedback submission."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(trace_output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    @patch('ncompass.trace.core.session.extract_source_code')
    def test_submit_feedback_success(self, mock_extract, mock_submit):
        """Test successful feedback submission."""
        mock_extract.return_value = "def test(): pass"
        mock_submit.return_value = {
            'targets': {
                'test.module': {'func_line_range_wrappings': []}
            }
        }
        
        result = self.session.submit_feedback(
            feedback_text="Why is this slow?",
            target_module="test.module",
            start_line=10,
            end_line=20
        )
        
        self.assertIn('targets', result)
        self.assertEqual(self.session.config_manager.iteration, 1)
        mock_submit.assert_called_once()
        
        # Verify feedback context was stored
        self.assertEqual(self.session.latest_feedback_context['feedback_text'], "Why is this slow?")
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    @patch('ncompass.trace.core.session.extract_source_code')
    def test_submit_feedback_with_trace_context(self, mock_extract, mock_submit):
        """Test feedback submission with trace context."""
        mock_extract.return_value = "def test(): pass"
        mock_submit.return_value = {'targets': {}}
        
        self.session.latest_trace_summary = {
            'markdown': 'Previous trace summary'
        }
        
        self.session.submit_feedback(
            feedback_text="Analyze this",
            target_module="test.module",
            start_line=1,
            end_line=10
        )
        
        call_kwargs = mock_submit.call_args[1]
        self.assertEqual(call_kwargs['request']['trace_context'], 'Previous trace summary')


class TestProfilingSessionConfig(unittest.TestCase):
    """Test cases for configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(
            trace_output_dir=self.temp_dir,
            cache_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_current_config(self):
        """Test get_current_config returns config manager's config."""
        config = self.session.get_current_config()
        self.assertEqual(config, {})
    
    def test_get_config_stats(self):
        """Test get_config_stats returns stats."""
        stats = self.session.get_config_stats()
        self.assertIn('iteration', stats)
        self.assertEqual(stats['iteration'], 0)
    
    def test_get_config_file_path_with_name(self):
        """Test get_config_file_path with explicit name."""
        path = self.session.get_config_file_path("custom_config")
        self.assertEqual(path, f"{self.temp_dir}/custom_config.json")
    
    def test_get_config_file_path_with_trace_name(self):
        """Test get_config_file_path uses trace name."""
        self.session.latest_trace_name = "test_trace_123"
        path = self.session.get_config_file_path()
        self.assertEqual(path, f"{self.temp_dir}/profile_config_test_trace_123.json")
    
    def test_get_config_file_path_default(self):
        """Test get_config_file_path default behavior."""
        path = self.session.get_config_file_path()
        self.assertEqual(path, f"{self.temp_dir}/profile_config.json")
    
    @patch('ncompass.trace.core.session.ConfigManager.save_to_file')
    def test_save_config(self, mock_save):
        """Test save_config calls config manager."""
        self.session.latest_trace_name = "test_trace"
        self.session.save_config()
        
        expected_path = f"{self.temp_dir}/profile_config_test_trace.json"
        mock_save.assert_called_once_with(expected_path)
    
    @patch('ncompass.trace.core.session.ConfigManager.load_from_file')
    def test_load_config(self, mock_load):
        """Test load_config calls config manager."""
        self.session.load_config("custom_config")
        
        expected_path = f"{self.temp_dir}/custom_config.json"
        mock_load.assert_called_once_with(expected_path)
    
    @patch('ncompass.trace.core.session.disable_rewrites')
    @patch('ncompass.trace.core.session.ConfigManager.reset')
    def test_reset(self, mock_config_reset, mock_disable):
        """Test reset clears session state."""
        self.session.latest_trace_path = "some/path"
        self.session.latest_trace_summary = {'data': 'test'}
        
        self.session.reset()
        
        self.assertIsNone(self.session.latest_trace_path)
        self.assertIsNone(self.session.latest_trace_summary)
        mock_config_reset.assert_called_once()
        mock_disable.assert_called_once()


class TestProfilingSessionTraceFiles(unittest.TestCase):
    """Test cases for trace file operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(trace_output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_find_latest_trace_success(self):
        """Test finding latest trace file."""
        # Create test trace files
        trace1 = Path(self.temp_dir) / "trace1.pt.trace.json"
        trace2 = Path(self.temp_dir) / "trace2.pt.trace.json"
        
        trace1.touch()
        import time
        time.sleep(0.01)
        trace2.touch()
        
        result = self.session._find_latest_trace()
        
        self.assertEqual(result, str(trace2))
    
    def test_find_latest_trace_no_files(self):
        """Test finding latest trace when no files exist."""
        with self.assertRaises(FileNotFoundError) as cm:
            self.session._find_latest_trace()
        
        self.assertIn("No trace files found", str(cm.exception))
    
    def test_find_and_rename_latest_trace(self):
        """Test finding and renaming latest trace."""
        # Create a test trace file
        original_trace = Path(self.temp_dir) / "original.pt.trace.json"
        original_trace.write_text('{"test": "data"}')
        
        result = self.session._find_and_rename_latest_trace("renamed_trace")
        
        expected_path = Path(self.temp_dir) / "renamed_trace.pt.trace.json"
        self.assertEqual(result, str(expected_path))
        self.assertTrue(expected_path.exists())
        self.assertFalse(original_trace.exists())
    
    def test_find_and_rename_gzipped_trace(self):
        """Test finding and renaming gzipped trace."""
        # Create a gzipped trace file
        gzipped_trace = Path(self.temp_dir) / "trace.pt.trace.json.gz"
        with gzip.open(gzipped_trace, 'wt') as f:
            f.write('{"test": "data"}')
        
        result = self.session._find_and_rename_latest_trace("decompressed_trace")
        
        expected_path = Path(self.temp_dir) / "decompressed_trace.pt.trace.json"
        self.assertEqual(result, str(expected_path))
        self.assertTrue(expected_path.exists())
        self.assertFalse(gzipped_trace.exists())
        
        # Verify content was decompressed
        content = expected_path.read_text()
        self.assertIn("test", content)


class TestProfilingSessionMarkers(unittest.TestCase):
    """Test cases for marker-related methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.session = ProfilingSession(trace_output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('ncompass.trace.core.session.enable_rewrites')
    @patch('ncompass.trace.core.session.RewriteConfig')
    def test_apply_targeted_markers(self, mock_config_class, mock_enable):
        """Test applying targeted markers."""
        mock_config = MagicMock()
        mock_config_class.from_dict.return_value = mock_config
        
        # Add some config first
        self.session.config_manager.add_config({'targets': {}}, merge=False)
        
        result = self.session.apply_targeted_markers()
        
        self.assertIn('targets', result)
        mock_enable.assert_called_once_with(config=mock_config)
    
    @patch('ncompass.trace.core.session.submit_queue_request')
    def test_filter_trace(self, mock_submit):
        """Test filter_trace method."""
        trace_path = f"{self.temp_dir}/test.pt.trace.json"
        filtered_path = f"{self.temp_dir}/test_filtered.pt.trace.json"
        
        self.session.latest_trace_path = trace_path
        mock_submit.return_value = filtered_path
        
        result = self.session.filter_trace(
            include_cuda_kernels=True,
            include_direct_children=False,
            min_duration_us=100.0
        )
        
        self.assertEqual(result, filtered_path)
        call_kwargs = mock_submit.call_args[1]
        filter_args = call_kwargs['request']['filter_args']
        self.assertTrue(filter_args['include_cuda_kernels'])
        self.assertFalse(filter_args['include_direct_children'])
        self.assertEqual(filter_args['min_duration_us'], 100.0)
    
    def test_filter_trace_no_trace(self):
        """Test filter_trace raises error when no trace available."""
        with self.assertRaises(ValueError) as cm:
            self.session.filter_trace()
        
        self.assertIn("No trace file available", str(cm.exception))


if __name__ == '__main__':
    unittest.main()