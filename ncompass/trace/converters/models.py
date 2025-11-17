"""Pydantic models for Chrome Trace events and conversion options."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class ChromeTraceEvent(BaseModel):
    """Chrome Trace event model with validation."""
    
    name: str = Field(..., description="Event name")
    ph: Literal["X", "B", "E", "i", "n", "O", "C", "b", "e", "s", "t", "f", "P", "N", "M", "c"] = Field(
        ..., description="Event phase"
    )
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

