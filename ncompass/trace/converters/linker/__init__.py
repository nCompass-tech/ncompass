"""Linker modules for connecting events via correlation IDs."""

from .nvtx_linker import link_nvtx_to_kernels
from .user_annotation_linker import link_user_annotation_to_kernels

__all__ = [
    "link_nvtx_to_kernels",
    "link_user_annotation_to_kernels",
]

