"""
Description: Torch profiler context managers for AST rewriting.
"""

import torch
from ncompass.trace.profile.base import _ProfileContextBase

class TorchRecordContext(_ProfileContextBase):
    """Context manager for Torch profiler record_function (for markers within a profile)."""
    def __init__(self, name: str):
        self.name = name
        self.context = torch.profiler.record_function(self.name)

    def __enter__(self):
        return self.context.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        return self.context.__exit__(exc_type, exc_value, traceback)