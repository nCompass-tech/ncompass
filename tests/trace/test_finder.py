"""
Tests for ncompass.trace.core.finder module.
"""

import unittest
import sys
import importlib.util
from types import ModuleType
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
        self.assertFalse(finder.ai_analysis_done)
    
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
    
    @patch('importlib.util.find_spec')
    def test_ai_analysis_enabled(self, mock_find_spec, mock_ai_analyzer_class):
        """Test AI analysis is triggered when USE_AI_PROFILING is enabled."""
        mock_analyzer = MagicMock()
        mock_ai_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_codebase.return_value = {
            'vllm.model_executor.models.llama': {
                'func_line_range_wrappings': []
            }
        }
        
        # Mock the meta_path to find specs
        mock_spec = MagicMock()
        mock_spec.origin = '/path/to/llama.py'
        mock_spec.has_location = True
        mock_find_spec.return_value = mock_spec
        
        mock_finder = MagicMock()
        mock_finder.find_spec.return_value = mock_spec
        
        original_meta_path = sys.meta_path.copy()
        sys.meta_path = [mock_finder]
        
        try:
            # Add ai_analysis_targets to config so AI analysis will find modules
            config_with_targets = {
                'targets': self.config['targets'],
                'ai_analysis_targets': ['vllm.model_executor.models.llama']
            }
            finder = RewritingFinder(config=config_with_targets)
            
            # Verify AI analyzer was called
            mock_ai_analyzer_class.assert_called_once()
            mock_analyzer.analyze_codebase.assert_called_once()
            
            # Verify the file_paths argument passed to analyze_codebase
            call_args = mock_analyzer.analyze_codebase.call_args
            file_paths = call_args[0][0]
            self.assertIn('vllm.model_executor.models.llama', file_paths)
            self.assertEqual(file_paths['vllm.model_executor.models.llama'], '/path/to/llama.py')
            
            # Verify target was added to target_fullnames
            self.assertIn('vllm.model_executor.models.llama', finder.target_fullnames)
        finally:
            sys.meta_path = original_meta_path
    
    
    @patch.dict('os.environ', {'USE_AI_PROFILING': 'true'})
    def test_ai_analysis_error_handling(self, mock_ai_analyzer_class):
        """Test that errors in AI analysis are caught and handled."""
        mock_ai_analyzer_class.side_effect = Exception("AI analysis failed")
        
        # Should not raise, just log the error
        try:
            finder = RewritingFinder(config=self.config)
            # Finder should still be created despite AI analysis error
            self.assertIsNotNone(finder)
            self.assertTrue(finder.ai_analysis_done)
            # Manual configs should still be present even if AI analysis fails
            self.assertEqual(finder.merged_configs, self.config['targets'])
            self.assertIn('vllm.model_executor.models.llama', finder.merged_configs)
        except Exception as e:
            self.fail(f"AI analysis error should be caught, but got: {e}")


if __name__ == '__main__':
    unittest.main()
