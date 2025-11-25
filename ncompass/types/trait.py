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

"""Trait base class combining Immutable and ABC for abstract immutable types."""

from abc import ABC

from ncompass.types.immutable import Immutable


class Trait(Immutable, ABC):
    """Abstract base class for immutable traits.
    
    Combines Immutable (preventing attribute changes) with ABC (abstract base class)
    for defining abstract interfaces with immutable implementations.
    """
    
    def __new__(cls, *args, **kwargs):
        # Pass no arguments to Immutable.__new__
        return Immutable.__new__(cls)

