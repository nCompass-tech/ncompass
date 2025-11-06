"""
Tests for ncompass.trace.core.utils module.
"""

import unittest
from unittest.mock import patch, MagicMock

from ncompass.trace.core.utils import (
    extract_source_code,
    extract_code_region,
    markers_overlap,
    merge_marker_configs,
    get_request_status,
    submit_queue_request
)


class TestExtractSourceCode(unittest.TestCase):
    """Test cases for extract_source_code function."""
    
    def test_extract_source_code_success(self):
        """Test extract_source_code with valid module."""
        # Use a known module
        source = extract_source_code('unittest')
        
        self.assertIsNotNone(source)
        self.assertIsInstance(source, str)
        self.assertGreater(len(source), 0)
    
    @patch('importlib.util.find_spec')
    def test_extract_source_code_module_not_found(self, mock_find_spec):
        """Test extract_source_code when module is not found."""
        mock_find_spec.return_value = None
        
        result = extract_source_code('nonexistent.module')
        
        self.assertIsNone(result)
    
    @patch('importlib.util.find_spec')
    def test_extract_source_code_no_origin(self, mock_find_spec):
        """Test extract_source_code when spec has no origin."""
        mock_spec = MagicMock()
        mock_spec.origin = None
        mock_find_spec.return_value = mock_spec
        
        result = extract_source_code('test.module')
        
        self.assertIsNone(result)
    
    @patch('importlib.util.find_spec')
    @patch('builtins.open', side_effect=FileNotFoundError())
    def test_extract_source_code_file_not_found(self, mock_file, mock_find_spec):
        """Test extract_source_code when file cannot be opened."""
        mock_spec = MagicMock()
        mock_spec.origin = '/nonexistent/file.py'
        mock_find_spec.return_value = mock_spec
        
        result = extract_source_code('test.module')
        
        self.assertIsNone(result)
    
    @patch('importlib.util.find_spec')
    def test_extract_source_code_exception(self, mock_find_spec):
        """Test extract_source_code handles exceptions."""
        mock_find_spec.side_effect = Exception("Test error")
        
        result = extract_source_code('test.module')
        
        self.assertIsNone(result)


class TestExtractCodeRegion(unittest.TestCase):
    """Test cases for extract_code_region function."""
    
    @patch('ncompass.trace.core.utils.extract_source_code')
    def test_extract_code_region_success(self, mock_extract):
        """Test extract_code_region with valid range."""
        mock_extract.return_value = "line 1\nline 2\nline 3\nline 4\nline 5"
        
        result = extract_code_region('test.module', 2, 4)
        
        self.assertIsNotNone(result)
        self.assertEqual(result, "line 2\nline 3\nline 4")
    
    @patch('ncompass.trace.core.utils.extract_source_code')
    def test_extract_code_region_module_not_found(self, mock_extract):
        """Test extract_code_region when module not found."""
        mock_extract.return_value = None
        
        result = extract_code_region('nonexistent.module', 1, 5)
        
        self.assertIsNone(result)
    
    @patch('ncompass.trace.core.utils.extract_source_code')
    def test_extract_code_region_start_before_beginning(self, mock_extract):
        """Test extract_code_region with start line before beginning."""
        mock_extract.return_value = "line 1\nline 2\nline 3"
        
        result = extract_code_region('test.module', -5, 2)
        
        # Should clamp to start of file
        self.assertEqual(result, "line 1\nline 2")
    
    @patch('ncompass.trace.core.utils.extract_source_code')
    def test_extract_code_region_end_after_file_end(self, mock_extract):
        """Test extract_code_region with end line after file end."""
        mock_extract.return_value = "line 1\nline 2\nline 3"
        
        result = extract_code_region('test.module', 2, 100)
        
        # Should clamp to end of file
        self.assertEqual(result, "line 2\nline 3")
    
    @patch('ncompass.trace.core.utils.extract_source_code')
    def test_extract_code_region_single_line(self, mock_extract):
        """Test extract_code_region for a single line."""
        mock_extract.return_value = "line 1\nline 2\nline 3"
        
        result = extract_code_region('test.module', 2, 2)
        
        self.assertEqual(result, "line 2")


class TestMarkersOverlap(unittest.TestCase):
    """Test cases for markers_overlap function."""
    
    def test_markers_overlap_no_overlap(self):
        """Test markers_overlap with non-overlapping markers."""
        marker1 = {'start_line': 1, 'end_line': 5}
        marker2 = {'start_line': 10, 'end_line': 15}
        
        self.assertFalse(markers_overlap(marker1, marker2))
        self.assertFalse(markers_overlap(marker2, marker1))
    
    def test_markers_overlap_adjacent(self):
        """Test markers_overlap with adjacent markers."""
        marker1 = {'start_line': 1, 'end_line': 5}
        marker2 = {'start_line': 6, 'end_line': 10}
        
        self.assertFalse(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker2_starts_within_marker1_extends_beyond(self):
        """Test markers_overlap when marker2 starts in marker1 but extends beyond."""
        marker1 = {'start_line': 1, 'end_line': 10}
        marker2 = {'start_line': 5, 'end_line': 15}
        
        self.assertTrue(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker1_starts_within_marker2_extends_beyond(self):
        """Test markers_overlap when marker1 starts in marker2 but extends beyond."""
        marker1 = {'start_line': 5, 'end_line': 15}
        marker2 = {'start_line': 1, 'end_line': 10}
        
        self.assertTrue(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker2_ends_at_marker1_start(self):
        """Test markers_overlap when marker2 ends exactly at marker1 start."""
        marker1 = {'start_line': 10, 'end_line': 20}
        marker2 = {'start_line': 5, 'end_line': 10}
        
        # Check the reverse: start1 >= start2 and start1 <= end2 and end1 > end2
        # 10 >= 5 (True), 10 <= 10 (True), 20 > 10 (True) -> True
        # So marker1 starts within marker2 but extends beyond - this IS an overlap
        self.assertTrue(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker2_within_marker1(self):
        """Test markers_overlap when marker2 is completely within marker1."""
        marker1 = {'start_line': 1, 'end_line': 20}
        marker2 = {'start_line': 5, 'end_line': 10}
        
        # marker2 is subset, not overlapping (based on the function logic)
        self.assertFalse(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker1_within_marker2(self):
        """Test markers_overlap when marker1 is completely within marker2."""
        marker1 = {'start_line': 5, 'end_line': 10}
        marker2 = {'start_line': 1, 'end_line': 20}
        
        # marker1 is subset, not overlapping
        self.assertFalse(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_exact_same_range(self):
        """Test markers_overlap with identical markers."""
        marker1 = {'start_line': 5, 'end_line': 10}
        marker2 = {'start_line': 5, 'end_line': 10}
        
        self.assertFalse(markers_overlap(marker1, marker2))
    
    def test_markers_overlap_marker2_starts_at_marker1_end(self):
        """Test markers_overlap when marker2 starts at marker1's end and extends."""
        marker1 = {'start_line': 1, 'end_line': 10}
        marker2 = {'start_line': 10, 'end_line': 15}
        
        # 10 >= 1 (True), 10 <= 10 (True), 15 > 10 (True) -> True
        self.assertTrue(markers_overlap(marker1, marker2))


class TestMergeMarkerConfigs(unittest.TestCase):
    """Test cases for merge_marker_configs function."""
    
    def test_merge_marker_configs_no_ai_configs(self):
        """Test merge_marker_configs with empty AI configs."""
        ai_configs = {}
        manual_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        self.assertEqual(result, manual_configs)
    
    def test_merge_marker_configs_no_manual_configs(self):
        """Test merge_marker_configs with empty manual configs."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        manual_configs = {}
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        self.assertEqual(result, ai_configs)
    
    def test_merge_marker_configs_different_files(self):
        """Test merge_marker_configs with different files."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        manual_configs = {
            'module2.py': {
                'func_line_range_wrappings': [
                    {'function': 'bar', 'start_line': 10, 'end_line': 15}
                ]
            }
        }
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        self.assertIn('module1.py', result)
        self.assertIn('module2.py', result)
    
    def test_merge_marker_configs_no_conflicts(self):
        """Test merge_marker_configs with same file but no conflicts."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        manual_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'bar', 'start_line': 10, 'end_line': 15}
                ]
            }
        }
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        # Both should be included
        self.assertEqual(len(result['module1.py']['func_line_range_wrappings']), 2)
    
    def test_merge_marker_configs_with_conflicts(self):
        """Test merge_marker_configs with overlapping markers."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 5, 'end_line': 15}  # Overlaps with manual
                ]
            }
        }
        manual_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'bar', 'start_line': 1, 'end_line': 10}  # Overlaps with AI
                ]
            }
        }
        
        with patch('ncompass.trace.infra.utils.logger.debug'):
            result = merge_marker_configs(ai_configs, manual_configs)
        
        # Only manual marker should be included
        self.assertEqual(len(result['module1.py']['func_line_range_wrappings']), 1)
        self.assertEqual(result['module1.py']['func_line_range_wrappings'][0]['function'], 'bar')
    
    def test_merge_marker_configs_multiple_ai_markers_some_conflict(self):
        """Test merge_marker_configs with multiple AI markers, some conflicting."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'ai_foo', 'start_line': 5, 'end_line': 15},  # Conflicts
                    {'function': 'ai_bar', 'start_line': 20, 'end_line': 25}  # No conflict
                ]
            }
        }
        manual_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'manual_foo', 'start_line': 1, 'end_line': 10}
                ]
            }
        }
        
        with patch('ncompass.trace.infra.utils.logger.debug'):
            result = merge_marker_configs(ai_configs, manual_configs)
        
        # Manual + one non-conflicting AI marker
        self.assertEqual(len(result['module1.py']['func_line_range_wrappings']), 2)
        functions = [m['function'] for m in result['module1.py']['func_line_range_wrappings']]
        self.assertIn('manual_foo', functions)
        self.assertIn('ai_bar', functions)
        self.assertNotIn('ai_foo', functions)
    
    def test_merge_marker_configs_no_wrappings_in_manual(self):
        """Test merge_marker_configs when manual config has no wrappings."""
        ai_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        manual_configs = {
            'module1.py': {
                'class_replacements': {'OldClass': 'NewClass'}
            }
        }
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        # AI wrappings should be added
        self.assertIn('func_line_range_wrappings', result['module1.py'])
        self.assertEqual(len(result['module1.py']['func_line_range_wrappings']), 1)
        # Manual class_replacements should be preserved
        self.assertIn('class_replacements', result['module1.py'])
    
    def test_merge_marker_configs_no_wrappings_in_ai(self):
        """Test merge_marker_configs when AI config has no wrappings."""
        ai_configs = {
            'module1.py': {
                'class_replacements': {'AIClass': 'NewClass'}
            }
        }
        manual_configs = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'foo', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        
        result = merge_marker_configs(ai_configs, manual_configs)
        
        # Manual wrappings should be preserved
        self.assertEqual(len(result['module1.py']['func_line_range_wrappings']), 1)


class TestGetRequestStatus(unittest.TestCase):
    """Test cases for get_request_status function."""
    
    @patch('requests.get')
    def test_get_request_status_success(self, mock_get):
        """Test get_request_status with successful request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'status': 'completed', 'result': 'data'}
        mock_get.return_value = mock_response
        
        result = get_request_status('request_123', 'http://localhost:8000')
        
        self.assertEqual(result, {'status': 'completed', 'result': 'data'})
        mock_get.assert_called_once_with('http://localhost:8000/status/request_123')
    
    @patch('requests.get')
    def test_get_request_status_different_base_url(self, mock_get):
        """Test get_request_status with different base URL."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'status': 'pending'}
        mock_get.return_value = mock_response
        
        result = get_request_status('req_456', 'https://example.com:9000')
        
        mock_get.assert_called_once_with('https://example.com:9000/status/req_456')


class TestSubmitQueueRequest(unittest.TestCase):
    """Test cases for submit_queue_request function."""
    
    @patch('requests.post')
    def test_submit_queue_request_no_await(self, mock_post):
        """Test submit_queue_request without awaiting result."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'request_id': 'req_123', 'status': 'pending'}
        mock_post.return_value = mock_response
        
        result = submit_queue_request(
            request={'data': 'test'},
            base_url='http://localhost:8000',
            endpoint='process',
            await_result=False
        )
        
        self.assertEqual(result, 'req_123')
        mock_post.assert_called_once_with(
            'http://localhost:8000/process',
            json={'data': 'test'}
        )
    
    @patch('requests.post')
    def test_submit_queue_request_no_request_id(self, mock_post):
        """Test submit_queue_request when no request_id in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'status': 'error'}
        mock_post.return_value = mock_response
        
        with self.assertRaises(ValueError) as cm:
            submit_queue_request(
                request={'data': 'test'},
                base_url='http://localhost:8000',
                endpoint='process',
                await_result=False
            )
        
        self.assertIn("Failed to submit request", str(cm.exception))
    
    @patch('ncompass.trace.core.utils.get_request_status')
    @patch('requests.post')
    @patch('time.sleep')
    def test_submit_queue_request_await_completed(self, mock_sleep, mock_post, mock_get_status):
        """Test submit_queue_request with await_result=True and completed status."""
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {
            'request_id': 'req_123',
            'status': 'pending'
        }
        mock_post.return_value = mock_post_response
        
        # Simulate status progression: pending -> completed
        mock_get_status.side_effect = [
            {'status': 'processing', 'request_id': 'req_123'},
            {'status': 'completed', 'result': {'data': 'final_result'}}
        ]
        
        result = submit_queue_request(
            request={'data': 'test'},
            base_url='http://localhost:8000',
            endpoint='analyze',
            await_result=True
        )
        
        self.assertEqual(result, {'data': 'final_result'})
        self.assertEqual(mock_get_status.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 2)  # Called for each status check
    
    @patch('ncompass.trace.core.utils.get_request_status')
    @patch('requests.post')
    @patch('time.sleep')
    def test_submit_queue_request_await_failed(self, mock_sleep, mock_post, mock_get_status):
        """Test submit_queue_request with await_result=True and failed status."""
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {
            'request_id': 'req_123',
            'status': 'pending'
        }
        mock_post.return_value = mock_post_response
        
        mock_get_status.return_value = {
            'status': 'failed',
            'error': 'Processing error'
        }
        
        with self.assertRaises(ValueError) as cm:
            submit_queue_request(
                request={'data': 'test'},
                base_url='http://localhost:8000',
                endpoint='analyze',
                await_result=True
            )
        
        self.assertIn("Request failed", str(cm.exception))
        self.assertIn("Processing error", str(cm.exception))
    
    @patch('ncompass.trace.core.utils.get_request_status')
    @patch('requests.post')
    @patch('time.sleep')
    def test_submit_queue_request_await_immediate_completion(self, mock_sleep, mock_post, mock_get_status):
        """Test submit_queue_request when request completes immediately."""
        mock_post_response = MagicMock()
        # Set up the response object to behave correctly
        mock_data = {
            'request_id': 'req_123',
            'status': 'completed',
            'result': {'immediate': 'result'}
        }
        mock_post_response.json.return_value = mock_data
        # Also make the response subscriptable for response['result']
        mock_post_response.__getitem__.side_effect = lambda key: mock_data[key]
        mock_post.return_value = mock_post_response
        
        result = submit_queue_request(
            request={'data': 'test'},
            base_url='http://localhost:8000',
            endpoint='analyze',
            await_result=True
        )
        
        # When initial status is 'completed', it returns response['result']
        # where response is the POST response object
        self.assertEqual(result, {'immediate': 'result'})
        # Should not call get_request_status since it's already completed  
        mock_get_status.assert_not_called()


if __name__ == '__main__':
    unittest.main()

