"""Pydantic models for Chrome Trace events and conversion options."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# All valid Chrome Trace event phases (excluding deprecated ones)
# Based on Chrome Trace Format spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
# 
# Event types:
# - Duration Events: B (begin), E (end)
# - Complete Events: X
# - Instant Events: i
# - Counter Events: C
# - Async Events: b (nestable start), n (nestable instant), e (nestable end)
# - Flow Events: s (start), t (step), f (end)
# - Sample Events: P
# - Object Events: N (created), O (snapshot), D (destroyed)
# - Metadata Events: M
# - Memory Dump Events: V (global), v (process)
# - Mark Events: R
# - Clock Sync Events: c
# - Context Events: ( (start), ) (end)
_CHROME_TRACE_PHASES_TUPLE = (
    # Duration Events
    "B", "E",
    # Complete Events
    "X",
    # Instant Events
    "i",
    # Counter Events
    "C",
    # Async Events
    "b", "n", "e",
    # Flow Events
    "s", "t", "f",
    # Sample Events
    "P",
    # Object Events
    "N", "O", "D",
    # Metadata Events
    "M",
    # Memory Dump Events
    "V", "v",
    # Mark Events
    "R",
    # Clock Sync Events
    "c",
    # Context Events
    "(", ")",
)

# Runtime set for validation (derived from tuple)
VALID_CHROME_TRACE_PHASES: set[str] = set(_CHROME_TRACE_PHASES_TUPLE)

# Type annotation for Pydantic - explicit Literal for Python 3.10 compatibility
# (Literal[*tuple] unpacking syntax requires Python 3.11+)
CHROME_TRACE_PHASES = Literal[
    # Duration Events
    "B", "E",
    # Complete Events
    "X",
    # Instant Events
    "i",
    # Counter Events
    "C",
    # Async Events
    "b", "n", "e",
    # Flow Events
    "s", "t", "f",
    # Sample Events
    "P",
    # Object Events
    "N", "O", "D",
    # Metadata Events
    "M",
    # Memory Dump Events
    "V", "v",
    # Mark Events
    "R",
    # Clock Sync Events
    "c",
    # Context Events
    "(", ")",
]


class ChromeTraceEvent(BaseModel):
    """Chrome Trace event model with validation."""
    
    name: str = Field(..., description="Event name")
    ph: CHROME_TRACE_PHASES = Field(..., description="Event phase")
    ts: float = Field(..., description="Timestamp in microseconds")
    pid: str = Field(..., description="Process ID (e.g., 'Device 0')")
    tid: str = Field(..., description="Thread ID (e.g., 'Stream 1')")
    cat: str = Field(..., description="Category (e.g., 'cuda', 'nvtx', 'osrt')")
    args: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    dur: Optional[float] = Field(None, description="Duration in microseconds (for 'X' events)")
    cname: Optional[str] = Field(None, description="Color name for visualization")
    id: Optional[int | str] = Field(None, description="Flow event ID for linking related events")
    bp: Optional[Literal["e", "s"]] = Field(None, description="Binding point for flow events: 'e' (enclosing) or 's' (same)")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = self.model_dump(exclude_none=True)
        return result


class ConversionOptions(BaseModel):
    """Configuration options for conversion."""
    
    activity_types: list[str] = Field(
        default=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
        description="Event types to include"
    )
    nvtx_event_prefix: Optional[list[str]] = Field(
        None, description="Filter NVTX events by name prefix"
    )
    nvtx_color_scheme: dict[str, str] = Field(
        default_factory=dict,
        description="Color mapping for NVTX events (regex -> color name)"
    )
    include_metadata: bool = Field(
        True, description="Include process/thread name metadata events"
    )

