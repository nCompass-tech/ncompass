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
nCompass Python SDK

For usage, import from submodules:
    from ncompass.trace import ProfilingSession, enable_rewrites
"""

from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read version from pyproject.toml
try:
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    __version__ = pyproject_data.get("project", {}).get("version", "unknown")
except (FileNotFoundError, KeyError, Exception):
    __version__ = "unknown"

