"""
Description: NVTX utils for AST rewriting.
"""


import torch
from ncompass.trace.infra.utils import tag
from ncompass.trace.profile.base import _ProfileContextBase

class NvtxContext(_ProfileContextBase):
    """Context manager for NVTX ranges."""
    def __init__(self, name: str):
        self.name = name
        self.tag_info = []

    def __enter__(self):
        torch.cuda.nvtx.range_push(tag(self.generate_tag_info()))

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.nvtx.range_pop()

    def generate_tag_info(self) -> list[str]:
        self.tag_info.append(f"name={self.name}")
        return self.tag_info