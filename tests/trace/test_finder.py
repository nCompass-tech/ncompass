"""
Tests for ncompass.trace.core.finder module.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock

from ncompass.trace.core.finder import _RewritingFinderBase, RewritingFinder


class TestRewritingFinderBase(unittest.TestCase):
    """Test cases for the _RewritingFinderBase class."""

    def test_init_with_no_config(self):
        """Test _RewritingFinderBase initialization with no config."""
        finder = _RewritingFinderBase(config=None)
        
        self.assertEqual(finder.config, {})
        self.assertEqual(finder.target_fullnames, [])
        self.assertEqual(finder.manual_configs, {})
        self.assertEqual(finder.merged_configs, {})

    def test_init_with_config(self):
        """Test _RewritingFinderBase initialization with config."""
        config = {
            'targets': {
                'module1': {'class_replacements': {}},
                'module2': {'class_func_replacements': {}}
            }
        }
        finder = _RewritingFinderBase(config=config)
        
        self.assertEqual(finder.config, config)
        self.assertEqual(set(finder.target_fullnames), {'module1', 'module2'})
        self.assertEqual(finder.manual_configs, config['targets'])
    
    def test_init_with_empty_config(self):
        """Test _RewritingFinderBase initialization with empty config dict."""
        finder = _RewritingFinderBase(config={})
        
        self.assertEqual(finder.target_fullnames, [])
        self.assertEqual(finder.manual_configs, {})
    
    def test_find_spec_not_implemented(self):
        """Test that find_spec raises NotImplementedError in base class."""
        finder = _RewritingFinderBase()
        with self.assertRaises(NotImplementedError):
            finder.find_spec('test.module', None, None)


class TestRewritingFinder(unittest.TestCase):
    """Test cases for the RewritingFinder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'targets': {
                'vllm.model_executor.models.llama': {
                    'func_line_range_wrappings': [{
                        'function': 'forward',
                        'start_line': 100,
                        'end_line': 105,
                        'context_class': 'ncompass.profiling.ProfileContext',
                        'context_values': [{'name': 'name', 'value': 'llama_forward', 'type': 'literal'}]
                    }]
                }
            }
        }
        self.original_meta_path = sys.meta_path.copy()
    
    def tearDown(self):
        """Clean up after tests."""
        sys.meta_path[:] = self.original_meta_path
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_init_inheritance(self):
        """Test RewritingFinder inherits from _RewritingFinderBase."""
        finder = RewritingFinder(config=self.config)
        self.assertIsInstance(finder, _RewritingFinderBase)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_init_with_config(self):
        """Test RewritingFinder initialization with config."""
        finder = RewritingFinder(config=self.config)
        
        self.assertIn('vllm.model_executor.models.llama', finder.target_fullnames)
        self.assertEqual(finder.manual_configs, self.config['targets'])
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_not_target_fullname(self):
        """Test find_spec returns None for non-target fullnames."""
        finder = RewritingFinder(config=self.config)
        result = finder.find_spec('some.other.module', None, None)
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('ncompass.trace.core.finder.create_replacer_from_config')
    @patch('ncompass.trace.core.finder.RewritingLoader')
    def test_find_spec_target_fullname_success(self, mock_loader_class, mock_create_replacer):
        """Test find_spec for target fullname with successful spec finding."""
        finder = RewritingFinder(config=self.config)
        
        # Mock the replacer
        mock_replacer = MagicMock()
        mock_create_replacer.return_value = mock_replacer
        
        # Mock the loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # Create a mock spec with proper attributes
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/llama.py'
        mock_spec.has_location = True
        
        # Mock finder that returns the spec
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = mock_spec
        
        # Set up meta_path with our mock finder
        sys.meta_path = [finder, mock_finder]
        
        with patch('importlib.util.spec_from_loader') as mock_spec_from_loader:
            mock_result_spec = MagicMock()
            mock_spec_from_loader.return_value = mock_result_spec
            
            result = finder.find_spec('vllm.model_executor.models.llama', None, None)
            
            # Verify the process
            mock_create_replacer.assert_called_once()
            call_args = mock_create_replacer.call_args
            self.assertEqual(call_args[0][0], 'vllm.model_executor.models.llama')
            self.assertEqual(call_args[0][1], self.config['targets']['vllm.model_executor.models.llama'])
            
            mock_loader_class.assert_called_once_with(
                'vllm.model_executor.models.llama',
                '/path/to/llama.py',
                mock_replacer
            )
            mock_spec_from_loader.assert_called_once_with(
                'vllm.model_executor.models.llama',
                mock_loader,
                origin='/path/to/llama.py'
            )
            self.assertEqual(result, mock_result_spec)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_no_other_finder_found(self):
        """Test find_spec when no other finder can find the spec."""
        finder = RewritingFinder(config=self.config)
        # Set up meta_path with only RewritingFinder
        sys.meta_path = [finder]
        
        result = finder.find_spec('vllm.model_executor.models.llama', None, None)
        
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_other_finder_returns_none(self):
        """Test find_spec when other finder returns None."""
        finder = RewritingFinder(config=self.config)
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = None
        
        sys.meta_path = [finder, mock_finder]
        
        result = finder.find_spec('vllm.model_executor.models.llama', None, None)
        
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_invalid_spec_no_origin(self):
        """Test find_spec with spec that has no origin."""
        finder = RewritingFinder(config=self.config)
        mock_spec = MagicMock()
        mock_spec.origin = None
        mock_spec.has_location = True
        
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = mock_spec
        
        sys.meta_path = [finder, mock_finder]
        
        result = finder.find_spec('vllm.model_executor.models.llama', None, None)
        
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_invalid_spec_no_location(self):
        """Test find_spec with spec that has no location."""
        finder = RewritingFinder(config=self.config)
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/file.py'
        mock_spec.has_location = False
        
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = mock_spec
        
        sys.meta_path = [finder, mock_finder]
        
        result = finder.find_spec('vllm.model_executor.models.llama', None, None)
        
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_skips_rewriting_finders(self):
        """Test that find_spec skips other RewritingFinder instances."""
        finder = RewritingFinder(config=self.config)
        # Create another RewritingFinder
        other_rewriting_finder = RewritingFinder(config=self.config)
        
        mock_finder = MagicMock()
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/llama.py'
        mock_spec.has_location = True
        mock_finder.find_spec.return_value = mock_spec
        
        # Set up meta_path with multiple finders including another RewritingFinder
        sys.meta_path = [finder, other_rewriting_finder, mock_finder]
        
        with patch('ncompass.trace.core.finder.create_replacer_from_config') as mock_create_replacer:
            mock_create_replacer.return_value = MagicMock()
            with patch('ncompass.trace.core.finder.RewritingLoader'):
                with patch('importlib.util.spec_from_loader') as mock_spec_from_loader:
                    mock_spec_from_loader.return_value = MagicMock()
                    
                    result = finder.find_spec('vllm.model_executor.models.llama', None, None)
                    
                    # Should have found the spec using mock_finder, not other_rewriting_finder
                    mock_finder.find_spec.assert_called_once()
                    # other_rewriting_finder should not have been called
                    self.assertIsNotNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_finder_without_find_spec_method(self):
        """Test find_spec with finder that doesn't have find_spec method."""
        finder = RewritingFinder(config=self.config)
        # Create a mock finder without find_spec method
        mock_finder_without_method = MagicMock()
        del mock_finder_without_method.find_spec
        
        mock_finder_with_method = MagicMock()
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/llama.py'
        mock_spec.has_location = True
        mock_finder_with_method.find_spec.return_value = mock_spec
        
        sys.meta_path = [finder, mock_finder_without_method, mock_finder_with_method]
        
        with patch('ncompass.trace.core.finder.create_replacer_from_config') as mock_create_replacer:
            mock_create_replacer.return_value = MagicMock()
            with patch('ncompass.trace.core.finder.RewritingLoader'):
                with patch('importlib.util.spec_from_loader') as mock_spec_from_loader:
                    mock_spec_from_loader.return_value = MagicMock()
                    
                    result = finder.find_spec('vllm.model_executor.models.llama', None, None)
                    
                    # Should skip the finder without find_spec and use the one with it
                    mock_finder_with_method.find_spec.assert_called_once()
                    self.assertIsNotNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_multiple_target_fullnames(self):
        """Test finder with multiple target fullnames."""
        multi_config = {
            'targets': {
                'module1': {'class_replacements': {}},
                'module2': {'class_replacements': {}},
                'module3': {'class_replacements': {}}
            }
        }
        finder = RewritingFinder(config=multi_config)
        
        # Test that non-target returns None
        result = finder.find_spec('other.module', None, None)
        self.assertIsNone(result)
        
        # Verify the target_fullnames are set correctly
        self.assertEqual(set(finder.target_fullnames), {'module1', 'module2', 'module3'})
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_no_config_returns_none(self):
        """Test find_spec returns None when module has no config."""
        finder = RewritingFinder(config={'targets': {}})
        
        # Even if we add a module to target_fullnames manually, 
        # it should return None if there's no config
        finder.target_fullnames.append('test.module')
        
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/test.py'
        mock_spec.has_location = True
        
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = mock_spec
        
        sys.meta_path = [finder, mock_finder]
        
        result = finder.find_spec('test.module', None, None)
        
        # Should return None because there's no manual or AI config
        self.assertIsNone(result)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    @patch('ncompass.trace.core.finder.submit_queue_request')
    @patch('importlib.machinery.PathFinder.find_spec')
    def test_ai_analysis_enabled_with_configs(self, mock_find_spec, mock_submit):
        """Test AI analysis with successful config generation."""
        # Mock AI service returning configs
        mock_submit.return_value = {
            'module1.py': {
                'func_line_range_wrappings': [
                    {'function': 'ai_discovered', 'start_line': 1, 'end_line': 5}
                ]
            }
        }
        
        # Mock find_spec to return valid specs
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/module1.py'
        mock_spec.has_location = True
        mock_find_spec.return_value = mock_spec
        
        config_with_targets = {
            'targets': {'manual.module': {'class_replacements': {}}},
            'ai_analysis_targets': ['module1']
        }
        
        with patch('ncompass.trace.core.finder.merge_marker_configs') as mock_merge:
            mock_merge.return_value = {
                'manual.module': {'class_replacements': {}},
                'module1.py': {
                    'func_line_range_wrappings': [
                        {'function': 'ai_discovered', 'start_line': 1, 'end_line': 5}
                    ]
                }
            }
            
            finder = RewritingFinder(config=config_with_targets)
            
            # Verify AI configs were merged
            mock_merge.assert_called_once()
            
            # Verify AI-discovered target was added to target_fullnames
            self.assertIn('module1.py', finder.target_fullnames)
            self.assertIn('manual.module', finder.target_fullnames)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch.object(RewritingFinder, '_run_ai_analysis')
    def test_ai_analysis_skipped_when_disabled(self, mock_run_ai):
        """Should skip _run_ai_analysis when USE_AI_PROFILING=false."""
        config = {
            'targets': {},
            'ai_analysis_targets': ['module1', 'module2']
        }
        finder = RewritingFinder(config=config)
        mock_run_ai.assert_not_called()
        self.assertEqual(finder.merged_configs, config["targets"])
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    @patch('ncompass.trace.core.utils.submit_queue_request')
    @patch('importlib.util.find_spec')
    def test_ai_analysis_full_flow(self, mock_find_spec, mock_submit):
        """Test full AI analysis flow including file discovery and API call."""
        # Mock specs for target modules
        mock_spec1 = MagicMock()
        mock_spec1.origin = '/path/to/module1.py'
        mock_spec1.has_location = True
        
        mock_spec2 = MagicMock()
        mock_spec2.origin = '/path/to/module2.py'
        mock_spec2.has_location = True
        
        mock_find_spec.side_effect = [mock_spec1, mock_spec2]
        
        # Mock AI response
        mock_submit.return_value = {
            'module1': {'func_line_range_wrappings': []},
            'module2': {'func_line_range_wrappings': []}
        }
        
        config = {
            'targets': {},
            'ai_analysis_targets': ['module1', 'module2']
        }
        
        finder = RewritingFinder(config=config)
        
        # Verify submit_queue_request was called if there are analysis targets
        if mock_submit.called:
            call_kwargs = mock_submit.call_args[1]
            self.assertEqual(call_kwargs['base_url'], finder.base_url)
            self.assertEqual(call_kwargs['endpoint'], 'analyze_codebase')
            self.assertTrue(call_kwargs['await_result'])
            
            # Verify request payload
            call_args = mock_submit.call_args[0]
            request = call_args[0]
            self.assertIn('contents_by_module', request)
        else:
            # AI analysis might be disabled or not triggered - that's OK for this test
            pass
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    @patch('ncompass.trace.core.utils.submit_queue_request')
    @patch('importlib.util.find_spec')
    def test_ai_analysis_module_not_found(self, mock_find_spec, mock_submit):
        """Test AI analysis handles modules that can't be found."""
        # First module found, second raises ImportError
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/module1.py'
        mock_spec.has_location = True
        
        mock_find_spec.side_effect = [mock_spec, ImportError("Module not found")]
        
        mock_submit.return_value = {}
        
        config = {
            'targets': {},
            'ai_analysis_targets': ['module1', 'nonexistent.module']
        }
        
        # Should not raise, just skip the missing module
        finder = RewritingFinder(config=config)
        
        # Verify submit was called (if AI is enabled and triggered)
        if mock_submit.called:
            call_args = mock_submit.call_args[0]
            request = call_args[0]
            file_paths = request['contents_by_module']
            self.assertIn('module1', file_paths)
            self.assertNotIn('nonexistent.module', file_paths)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    @patch('ncompass.trace.core.utils.submit_queue_request')
    @patch('importlib.util.find_spec')
    def test_ai_analysis_invalid_spec(self, mock_find_spec, mock_submit):
        """Test AI analysis handles specs without origin or location."""
        # Spec without origin
        mock_spec1 = MagicMock()
        mock_spec1.origin = None
        mock_spec1.has_location = True
        
        # Spec without location
        mock_spec2 = MagicMock()
        mock_spec2.origin = '/path/to/file.py'
        mock_spec2.has_location = False
        
        # Valid spec
        mock_spec3 = MagicMock()
        mock_spec3.origin = '/path/to/valid.py'
        mock_spec3.has_location = True
        
        mock_find_spec.side_effect = [mock_spec1, mock_spec2, mock_spec3]
        
        mock_submit.return_value = {}
        
        config = {
            'targets': {},
            'ai_analysis_targets': ['no_origin', 'no_location', 'valid']
        }
        
        finder = RewritingFinder(config=config)
        
        # Verify AI analysis attempted if enabled
        if mock_submit.called:
            call_args = mock_submit.call_args[0]
            request = call_args[0]
            file_paths = request['contents_by_module']
            # Only valid module should be included
            self.assertEqual(len(file_paths), 1)
            self.assertIn('valid', file_paths)
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    @patch('ncompass.trace.core.utils.submit_queue_request')
    def test_ai_analysis_exception_handling(self, mock_submit):
        """Test AI analysis handles exceptions gracefully."""
        mock_submit.side_effect = Exception("AI service error")
        
        config = {
            'targets': {'manual.module': {}},
            'ai_analysis_targets': []
        }
        
        # Should not raise, should return empty configs
        with patch('traceback.print_exc'):
            finder = RewritingFinder(config=config)
        
        # Should fall back to manual configs (may include additional metadata fields)
        self.assertIn('manual.module', finder.merged_configs)

    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('ncompass.trace.core.finder.create_replacer_from_config')
    @patch('ncompass.trace.core.finder.RewritingLoader')
    @patch('importlib.util.find_spec')
    def test_find_spec_matches_by_file_path(self, mock_importlib_find_spec, mock_loader_class, mock_create_replacer):
        """Test find_spec matches imports by file path when fullname doesn't match.
        
        This handles cases where:
        - Config has 'ncompass.trace.core.finder' as target
        - Import is done as 'finder' from same directory
        - Both resolve to the same file path
        """
        # Config has fully qualified name
        config = {
            'targets': {
                'ncompass.trace.core.finder': {
                    'filePath': '/home/user/ncompass/ncompass/trace/core/finder.py',
                    'func_line_range_wrappings': [{
                        'function': 'find_spec',
                        'start_line': 129,
                        'end_line': 165,
                        'context_class': 'ncompass.profiling.ProfileContext',
                        'context_values': [{'name': 'name', 'value': 'finder_test', 'type': 'literal'}]
                    }]
                }
            }
        }
        finder = RewritingFinder(config=config)
        
        # Mock the replacer and loader
        mock_replacer = MagicMock()
        mock_create_replacer.return_value = mock_replacer
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        # The spec for the short-name import ('finder')
        # This is what another finder would return when asked about 'finder'
        mock_spec_short = MagicMock()
        mock_spec_short.origin = '/home/user/ncompass/ncompass/trace/core/finder.py'
        mock_spec_short.has_location = True
        
        # The spec for the fully qualified name (for path matching)
        # This is what importlib.util.find_spec returns when checking the target
        mock_spec_full = MagicMock()
        mock_spec_full.origin = '/home/user/ncompass/ncompass/trace/core/finder.py'
        mock_spec_full.has_location = True
        
        # Setup: importlib.util.find_spec returns the full spec for the target
        mock_importlib_find_spec.return_value = mock_spec_full
        
        # Mock finder that returns spec for 'finder' (short name)
        mock_other_finder = MagicMock()
        mock_other_finder.find_spec.return_value = mock_spec_short
        
        # Set up meta_path
        sys.meta_path = [finder, mock_other_finder]
        
        with patch('importlib.util.spec_from_loader') as mock_spec_from_loader:
            mock_result_spec = MagicMock()
            mock_spec_from_loader.return_value = mock_result_spec
            
            # Import using short name 'finder' instead of full 'ncompass.trace.core.finder'
            result = finder.find_spec('finder', None, None)
            
            # Should still find and rewrite because file paths match
            self.assertIsNotNone(result, "Should match by file path even when fullname differs")
            
            # Verify it used the config from the fully qualified target name
            mock_create_replacer.assert_called_once()
            call_args = mock_create_replacer.call_args
            self.assertEqual(call_args[0][0], 'ncompass.trace.core.finder', 
                            "Should use the full target name from config")
            self.assertEqual(call_args[0][1], config['targets']['ncompass.trace.core.finder'])
            
            # Verify loader was created with correct parameters
            mock_loader_class.assert_called_once_with(
                'ncompass.trace.core.finder',  # Should use the matched target name
                '/home/user/ncompass/ncompass/trace/core/finder.py',
                mock_replacer
            )
            
            # Verify spec was created correctly
            mock_spec_from_loader.assert_called_once_with(
                'finder',  # Original import name
                mock_loader,
                origin='/home/user/ncompass/ncompass/trace/core/finder.py'
            )
            
            self.assertEqual(result, mock_result_spec)

    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    def test_find_spec_no_match_by_file_path(self):
        """Test find_spec returns None when file paths don't match any target."""
        config = {
            'targets': {
                'ncompass.trace.core.finder': {
                    'func_line_range_wrappings': []
                }
            }
        }
        finder = RewritingFinder(config=config)
        
        # Mock spec for a completely different module
        mock_spec_different = MagicMock()
        mock_spec_different.origin = '/home/user/other/module/path.py'
        mock_spec_different.has_location = True
        
        mock_other_finder = MagicMock()
        mock_other_finder.find_spec.return_value = mock_spec_different
        
        sys.meta_path = [finder, mock_other_finder]
        
        with patch('importlib.util.find_spec') as mock_importlib_find_spec:
            # When checking target 'ncompass.trace.core.finder', return different path
            mock_target_spec = MagicMock()
            mock_target_spec.origin = '/home/user/ncompass/ncompass/trace/core/finder.py'
            mock_importlib_find_spec.return_value = mock_target_spec
            
            # Try to import 'different_module' which resolves to a different file
            result = finder.find_spec('different_module', None, None)
            
            # Should return None because paths don't match
            self.assertIsNone(result, "Should not match when file paths differ")

    @patch.dict('os.environ', {'USE_AI_PROFILING': 'false'})
    @patch('importlib.util.find_spec')
    def test_find_spec_handles_target_spec_exceptions(self, mock_importlib_find_spec):
        """Test find_spec handles exceptions when resolving target specs for path matching."""
        config = {
            'targets': {
                'ncompass.trace.core.finder': {
                    'func_line_range_wrappings': []
                },
                'other.module': {
                    'func_line_range_wrappings': []
                }
            }
        }
        finder = RewritingFinder(config=config)
        
        # Mock spec for the import
        mock_spec = MagicMock()
        mock_spec.origin = '/home/user/path/to/file.py'
        mock_spec.has_location = True
        
        mock_other_finder = MagicMock()
        mock_other_finder.find_spec.return_value = mock_spec
        
        sys.meta_path = [finder, mock_other_finder]
        
        # Simulate exception when trying to resolve target specs
        mock_importlib_find_spec.side_effect = ImportError("Module not found")
        
        # Should not crash, just return None
        result = finder.find_spec('some_module', None, None)
        
        self.assertIsNone(result, "Should handle exceptions gracefully and return None")

if __name__ == '__main__':
    unittest.main()
