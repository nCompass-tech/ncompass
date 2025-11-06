"""
Tests for ncompass.trace.infra.utils module.
"""

import unittest

from ncompass.trace.infra.utils import tag, deep_merge


class TestTag(unittest.TestCase):
    """Test cases for tag function."""
    
    def test_tag_single_string(self):
        """Test tag with single string."""
        result = tag("test_info")
        self.assertEqual(result, "[NC_TAG: test_info]")
    
    def test_tag_empty_string(self):
        """Test tag with empty string."""
        result = tag("")
        self.assertEqual(result, "[NC_TAG: ]")
    
    def test_tag_string_with_spaces(self):
        """Test tag with string containing spaces."""
        result = tag("test info with spaces")
        self.assertEqual(result, "[NC_TAG: test info with spaces]")
    
    def test_tag_list_single_element(self):
        """Test tag with list containing single element."""
        result = tag(["item1"])
        self.assertEqual(result, "[NC_TAG: item1]")
    
    def test_tag_list_multiple_elements(self):
        """Test tag with list containing multiple elements."""
        result = tag(["item1", "item2", "item3"])
        self.assertEqual(result, "[NC_TAG: item1][NC_TAG: item2][NC_TAG: item3]")
    
    def test_tag_empty_list(self):
        """Test tag with empty list."""
        result = tag([])
        self.assertEqual(result, "")
    
    def test_tag_list_with_empty_strings(self):
        """Test tag with list containing empty strings."""
        result = tag(["", "valid", ""])
        self.assertEqual(result, "[NC_TAG: ][NC_TAG: valid][NC_TAG: ]")
    
    def test_tag_list_with_numeric_strings(self):
        """Test tag with list containing numeric strings."""
        result = tag(["1", "2", "3"])
        self.assertEqual(result, "[NC_TAG: 1][NC_TAG: 2][NC_TAG: 3]")


class TestDeepMerge(unittest.TestCase):
    """Test cases for deep_merge function."""
    
    def test_deep_merge_different_types_override_wins(self):
        """Test deep_merge when base and override are different types."""
        base = {"key": "value"}
        override = "string"
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, "string")
    
    def test_deep_merge_dict_basic(self):
        """Test deep_merge with basic dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"c": 3, "d": 4}
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3, "d": 4})
    
    def test_deep_merge_dict_overlapping_keys(self):
        """Test deep_merge with overlapping keys."""
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, {"a": 1, "b": 99, "c": 3})
    
    def test_deep_merge_dict_nested(self):
        """Test deep_merge with nested dictionaries."""
        base = {
            "level1": {
                "level2a": {"key1": "value1"},
                "level2b": "value2"
            }
        }
        override = {
            "level1": {
                "level2a": {"key2": "value2"},
                "level2c": "value3"
            }
        }
        
        result = deep_merge(base, override)
        
        # level2a should have both key1 and key2
        self.assertIn("key1", result["level1"]["level2a"])
        self.assertIn("key2", result["level1"]["level2a"])
        # level2b from base should be preserved
        self.assertEqual(result["level1"]["level2b"], "value2")
        # level2c from override should be added
        self.assertEqual(result["level1"]["level2c"], "value3")
    
    def test_deep_merge_dict_deeply_nested(self):
        """Test deep_merge with deeply nested dictionaries."""
        base = {
            "a": {
                "b": {
                    "c": {
                        "d": "base_value"
                    }
                }
            }
        }
        override = {
            "a": {
                "b": {
                    "c": {
                        "e": "override_value"
                    }
                }
            }
        }
        
        result = deep_merge(base, override)
        
        self.assertEqual(result["a"]["b"]["c"]["d"], "base_value")
        self.assertEqual(result["a"]["b"]["c"]["e"], "override_value")
    
    def test_deep_merge_list_concatenate(self):
        """Test deep_merge with lists."""
        base = [1, 2, 3]
        override = [4, 5]
        
        result = deep_merge(base, override)
        
        # Should be override + base (without duplicates from override)
        self.assertEqual(result, [4, 5, 1, 2, 3])
    
    def test_deep_merge_list_with_duplicates(self):
        """Test deep_merge with lists containing duplicates."""
        base = [1, 2, 3, 4]
        override = [3, 4, 5, 6]
        
        result = deep_merge(base, override)
        
        # Should be override + (base items not in override)
        self.assertEqual(result, [3, 4, 5, 6, 1, 2])
    
    def test_deep_merge_list_empty_base(self):
        """Test deep_merge with empty base list."""
        base = []
        override = [1, 2, 3]
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, [1, 2, 3])
    
    def test_deep_merge_list_empty_override(self):
        """Test deep_merge with empty override list."""
        base = [1, 2, 3]
        override = []
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, [1, 2, 3])
    
    def test_deep_merge_string_override(self):
        """Test deep_merge with strings."""
        base = "base_string"
        override = "override_string"
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, "override_string")
    
    def test_deep_merge_int_override(self):
        """Test deep_merge with integers."""
        base = 42
        override = 99
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, 99)
    
    def test_deep_merge_bool_override(self):
        """Test deep_merge with booleans."""
        base = True
        override = False
        
        result = deep_merge(base, override)
        
        self.assertEqual(result, False)
    
    def test_deep_merge_none_values(self):
        """Test deep_merge with None values."""
        base = {"key": "value"}
        override = {"key": None}
        
        result = deep_merge(base, override)
        
        self.assertIsNone(result["key"])
    
    def test_deep_merge_doesnt_modify_inputs(self):
        """Test that deep_merge doesn't modify input dictionaries."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        
        result = deep_merge(base, override)
        
        # Modify result
        result["a"]["d"] = 3
        
        # Base and override should not be modified
        self.assertNotIn("d", base["a"])
        self.assertNotIn("d", override["a"])
    
    def test_deep_merge_complex_nested_structure(self):
        """Test deep_merge with complex nested structure."""
        base = {
            "targets": {
                "module1": {
                    "class_replacements": {"A": "B"},
                    "wrappings": [{"line": 1}]
                }
            },
            "settings": {
                "enabled": True
            }
        }
        override = {
            "targets": {
                "module1": {
                    "class_func_replacements": {"C": "D"},
                    "wrappings": [{"line": 2}]
                },
                "module2": {
                    "class_replacements": {"E": "F"}
                }
            },
            "settings": {
                "timeout": 30
            }
        }
        
        result = deep_merge(base, override)
        
        # Check module1 merged properly
        self.assertIn("class_replacements", result["targets"]["module1"])
        self.assertIn("class_func_replacements", result["targets"]["module1"])
        # Wrappings should be merged (override + base without duplicates)
        self.assertEqual(len(result["targets"]["module1"]["wrappings"]), 2)
        
        # Check module2 added
        self.assertIn("module2", result["targets"])
        
        # Check settings merged
        self.assertTrue(result["settings"]["enabled"])
        self.assertEqual(result["settings"]["timeout"], 30)
    
    def test_deep_merge_list_of_dicts(self):
        """Test deep_merge with lists of dictionaries."""
        base = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        override = [{"id": 3, "name": "c"}]
        
        result = deep_merge(base, override)
        
        # Should concatenate: override + base items not in override
        # Since dicts are compared by reference, all should be included
        self.assertEqual(len(result), 3)


if __name__ == '__main__':
    unittest.main()

