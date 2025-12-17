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
Trace validation utilities for integration tests.

This module provides utilities for parsing and validating trace files.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def count_events_by_category_json(
    trace_path: Path, 
    category: str,
    phase: str | None = "X"
) -> int:
    """
    Count events by category in a gzip-compressed Chrome trace JSON file.
    
    Args:
        trace_path: Path to .json.gz trace file (must be gzip compressed)
        category: Event category to count (e.g., "kernel")
        phase: Event phase to filter (e.g., "X" for complete events), 
               or None for all phases
    
    Returns:
        Number of matching events
        
    Raises:
        ValueError: If the file is not gzip compressed
    """
    # Validate gzip format
    if not str(trace_path).endswith(".gz"):
        raise ValueError(
            f"Only gzip-compressed files are supported. "
            f"Expected .json.gz, got: {trace_path}"
        )
    
    # Load trace data
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    # Count matching events
    events = data.get("traceEvents", [])
    count = 0
    for event in events:
        if event.get("cat") == category:
            if phase is None or event.get("ph") == phase:
                count += 1
    
    return count

