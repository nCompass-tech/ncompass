"""
Description: Loader for AST rewriting.
"""

import ast, importlib.abc

from ncompasslib.trait import Trait


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

    def get_filename(self, fullname):
        return self.path

    def get_data(self, path):
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