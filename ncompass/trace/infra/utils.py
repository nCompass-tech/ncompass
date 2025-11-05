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
Description: Utils for AST rewriting.
"""

from typing import Union, Any
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)
logger.propagate = False
handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='[%(asctime)s] (%(levelname)s) (%(filename)s:%(lineno)d): %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
for h in logger.handlers[:]:
    logger.removeHandler(h)
logger.addHandler(handler)

logger.info(f"NC Logger initialized")

def tag(info: Union[str, list[str]]) -> str:
    """Produce a formatted tag to annotate the trace."""
    if isinstance(info, list):
        return "".join([f"[NC_TAG: {i}]" for i in info])
    else:
        return f"[NC_TAG: {info}]"

def deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge two nested structures, with override taking precedence.
    """
    # If override is not the same type as base, override wins
    if type(base) != type(override):
        return deepcopy(override)
    
    # Handle dict merging
    if isinstance(base, dict):
        result = deepcopy(base)
        for key, value in override.items():
            if key in result:
                # Recursively merge if both values exist
                result[key] = deep_merge(result[key], value)
            else:
                # Add new key from override
                result[key] = deepcopy(value)
        return result
    
    # Handle list merging (concatenate)
    elif isinstance(base, list):
        return deepcopy(override) + [t for t in base if t not in override]
    
    # For other types (str, int, bool, etc.), override wins
    else:
        return deepcopy(override)