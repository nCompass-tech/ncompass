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
Description: Loader for AST rewriting.
"""

import ast
import importlib.abc

from ncompass.types.trait import Trait


class _RewritingLoader(Trait, importlib.abc.SourceLoader):
    """Base class for AST rewriting loaders."""
    def __init__(self, fullname, path, replacer):
        """
        Args:
            fullname: eg. vllm.model_executor.models.llama
            path: eg. /path/to/vllm/model_executor/models/llama.py
            replacer: eg. DynamicReplacer object
        """
        self.fullname = fullname
        self.path = path
        self.replacer = replacer

    def get_filename(self, fullname: str) -> str:
        """Get the filename for a module."""
        return self.path

    def get_data(self, path: str) -> bytes:
        """Read file data as bytes."""
        return open(path, "rb").read()
    
    def source_to_code(self, data, path, *, _optimize=-1):
        raise NotImplementedError

class RewritingLoader(_RewritingLoader):
    """Loader for AST rewriting. Targets a specific file."""
    
    def source_to_code(self, data, path, *, _optimize=-1):
        tree = ast.parse(data, filename=path)
        tree = self.replacer.visit(tree)
        ast.fix_missing_locations(tree)
        return compile(tree, path, "exec", dont_inherit=True, optimize=_optimize)
