"""Linker modules for connecting events via correlation IDs."""

from .nvtx_linker import link_nvtx_to_kernels
from .user_annotation_linker import link_user_annotation_to_kernels
from .nvtx_sql_linker import (
    can_use_sql_linking,
    stream_nvtx_kernel_events,
    stream_flow_events,
    get_mapped_nvtx_identifiers,
    stream_unmapped_nvtx_events,
)

__all__ = [
    # Python-based linking (loads all events into memory)
    "link_nvtx_to_kernels",
    "link_user_annotation_to_kernels",
    # SQL-based linking (streaming, memory-efficient)
    "can_use_sql_linking",
    "stream_nvtx_kernel_events",
    "stream_flow_events",
    "get_mapped_nvtx_identifiers",
    "stream_unmapped_nvtx_events",
]

