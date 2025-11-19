#!/usr/bin/env python3
"""
Helper script to compare generated files against gold files.
This script runs inside Docker containers to perform file comparisons.

Usage:
    python compare_files.py <generated_file> <gold_file>
    
Exit codes:
    0: Files match
    1: Files differ or error occurred
"""

import json
import sys
from pathlib import Path
from typing import Any

try:
    from deepdiff import DeepDiff
except ImportError:
    # Fallback to basic comparison if deepdiff is not available
    DeepDiff = None


def normalize_json_data(data: Any) -> Any:
    """
    Normalize JSON data by removing or normalizing non-deterministic fields.
    
    This includes:
    - Timestamps
    - UUIDs
    - File paths that may differ
    - Other non-deterministic fields
    """
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            # Skip or normalize non-deterministic fields
            if key in ["ts", "timestamp", "time", "startTime", "endTime"]:
                # Keep structure but normalize timestamp to 0 for comparison
                normalized[key] = 0
            elif key in ["id", "uuid", "pid", "tid"]:
                # Keep structure but normalize IDs
                normalized[key] = 0
            elif key in ["name", "cat"]:
                # Keep these as they're important for comparison
                normalized[key] = value
            elif isinstance(value, (dict, list)):
                normalized[key] = normalize_json_data(value)
            else:
                normalized[key] = value
        return normalized
    elif isinstance(data, list):
        return [normalize_json_data(item) for item in data]
    else:
        return data


def compare_json_files(generated_path: Path, gold_path: Path) -> tuple[bool, str]:
    """
    Compare two JSON files, handling non-deterministic fields.
    
    Returns:
        (match: bool, message: str)
    """
    try:
        with open(generated_path, "r") as f:
            generated_data = json.load(f)
        
        with open(gold_path, "r") as f:
            gold_data = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"Failed to parse JSON: {e}"
    except Exception as e:
        return False, f"Error reading files: {e}"
    
    # Normalize data to remove non-deterministic fields
    normalized_generated = normalize_json_data(generated_data)
    normalized_gold = normalize_json_data(gold_data)
    
    if DeepDiff is not None:
        # Use deepdiff for detailed comparison
        diff = DeepDiff(normalized_gold, normalized_generated, ignore_order=False, verbose_level=2)
        if not diff:
            return True, "Files match"
        else:
            return False, f"Files differ:\n{diff.pretty()}"
    else:
        # Fallback to simple comparison
        if normalized_generated == normalized_gold:
            return True, "Files match"
        else:
            # Try to find key differences
            if isinstance(normalized_generated, dict) and isinstance(normalized_gold, dict):
                gen_keys = set(normalized_generated.keys())
                gold_keys = set(normalized_gold.keys())
                missing_keys = gold_keys - gen_keys
                extra_keys = gen_keys - gold_keys
                diff_msg = []
                if missing_keys:
                    diff_msg.append(f"Missing keys in generated: {missing_keys}")
                if extra_keys:
                    diff_msg.append(f"Extra keys in generated: {extra_keys}")
                if diff_msg:
                    return False, "\n".join(diff_msg)
            return False, "Files differ (use deepdiff for detailed comparison)"


def compare_files_binary(generated_path: Path, gold_path: Path) -> tuple[bool, str]:
    """Compare two binary files."""
    try:
        with open(generated_path, "rb") as f:
            generated_data = f.read()
        
        with open(gold_path, "rb") as f:
            gold_data = f.read()
        
        if generated_data == gold_data:
            return True, "Files match"
        else:
            return False, f"Files differ (generated: {len(generated_data)} bytes, gold: {len(gold_data)} bytes)"
    except Exception as e:
        return False, f"Error reading files: {e}"


def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python compare_files.py <generated_file> <gold_file>", file=sys.stderr)
        sys.exit(1)
    
    generated_path = Path(sys.argv[1])
    gold_path = Path(sys.argv[2])
    
    if not generated_path.exists():
        print(f"Error: Generated file does not exist: {generated_path}", file=sys.stderr)
        sys.exit(1)
    
    if not gold_path.exists():
        print(f"Error: Gold file does not exist: {gold_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine file type and compare
    if generated_path.suffix == ".json" and gold_path.suffix == ".json":
        match, message = compare_json_files(generated_path, gold_path)
    else:
        # Binary comparison
        match, message = compare_files_binary(generated_path, gold_path)
    
    if match:
        print(f"✓ {message}")
        sys.exit(0)
    else:
        print(f"✗ {message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

