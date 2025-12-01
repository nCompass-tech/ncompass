#!/usr/bin/env python3
"""
Helper script to compare generated files against gold files.
This script runs inside Docker containers to perform file comparisons.

Usage:
    python compare.py <generated_file> <gold_file>
    
Exit codes:
    0: Files match
    1: Files differ or error occurred
"""

import gzip
import json
import sys
from pathlib import Path
from typing import Any

try:
    from deepdiff import DeepDiff
except ImportError:
    # Fallback to basic comparison if deepdiff is not available
    DeepDiff = None


def load_json_file(path: Path) -> Any:
    """
    Load JSON data from a file, handling both .json and .json.gz formats.
    
    Args:
        path: Path to the JSON file (can be .json or .json.gz)
        
    Returns:
        Parsed JSON data
    """
    if path.suffix == ".gz" or str(path).endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def normalize_json_data(data: Any) -> Any:
    """
    Normalize JSON data by removing or normalizing non-deterministic fields.
    
    This includes:
    - Timestamps (ts, timestamp, time, startTime, endTime)
    - Durations (dur)
    - IDs (id, uuid, pid, tid)
    - Other non-deterministic fields
    """
    if isinstance(data, dict):
        normalized = {}
        for key, value in data.items():
            # Skip or normalize non-deterministic fields
            if key in ["ts", "timestamp", "time", "startTime", "endTime", "dur"]:
                # Keep structure but normalize timestamp/duration to 0 for comparison
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


def sort_trace_events(data: Any) -> Any:
    """
    Sort traceEvents array by (name, cat) for deterministic ordering.
    
    Chrome trace format has a top-level 'traceEvents' array that may have
    events in non-deterministic order. This function sorts them for stable comparison.
    
    Args:
        data: JSON data (expected to be a dict with 'traceEvents' key)
        
    Returns:
        Data with sorted traceEvents
    """
    if not isinstance(data, dict):
        return data
    
    result = data.copy()
    
    if "traceEvents" in result and isinstance(result["traceEvents"], list):
        # Sort by (name, cat) tuple, handling missing keys
        def sort_key(event: Any) -> tuple:
            if not isinstance(event, dict):
                return ("", "")
            name = event.get("name", "") or ""
            cat = event.get("cat", "") or ""
            return (name, cat)
        
        result["traceEvents"] = sorted(result["traceEvents"], key=sort_key)
    
    return result


def compare_json_files(generated_path: Path, gold_path: Path) -> tuple[bool, str]:
    """
    Compare two JSON files, handling non-deterministic fields.
    
    Supports both .json and .json.gz files. Normalizes timestamps, durations,
    and IDs, and sorts traceEvents for deterministic comparison.
    
    Returns:
        (match: bool, message: str)
    """
    try:
        generated_data = load_json_file(generated_path)
        gold_data = load_json_file(gold_path)
    except json.JSONDecodeError as e:
        return False, f"Failed to parse JSON: {e}"
    except Exception as e:
        return False, f"Error reading files: {e}"
    
    # Normalize data to remove non-deterministic fields
    normalized_generated = normalize_json_data(generated_data)
    normalized_gold = normalize_json_data(gold_data)
    
    # Sort traceEvents for deterministic comparison
    normalized_generated = sort_trace_events(normalized_generated)
    normalized_gold = sort_trace_events(normalized_gold)
    
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
        print("Usage: python compare.py <generated_file> <gold_file>", file=sys.stderr)
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
    # Check for JSON files (including .json.gz)
    gen_is_json = (generated_path.suffix == ".json" or 
                   str(generated_path).endswith(".json.gz"))
    gold_is_json = (gold_path.suffix == ".json" or 
                    str(gold_path).endswith(".json.gz"))
    
    if gen_is_json and gold_is_json:
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

