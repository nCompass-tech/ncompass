"""Adapters for different event formats."""

import logging
from typing import Any, Optional
from ncompass.types import Trait
from ..models import ChromeTraceEvent

logger = logging.getLogger(__name__)


class EventAdapter(Trait):
    """Abstract base class for event format adapters."""
    
    def get_time_range(self, event: Any) -> Optional[tuple[float, float]]:
        """Extract time range from event.
        
        Args:
            event: Event in the adapter's format
            
        Returns:
            Tuple of (start, end) time or None if invalid
        """
        raise NotImplementedError
    
    def get_correlation_id(self, event: Any) -> Optional[int]:
        """Get correlation ID from event.
        
        Args:
            event: Event in the adapter's format
            
        Returns:
            Correlation ID or None
        """
        raise NotImplementedError
    
    def get_event_id(self, event: Any) -> tuple:
        """Get unique identifier for event.
        
        Args:
            event: Event in the adapter's format
            
        Returns:
            Tuple that uniquely identifies the event
        """
        raise NotImplementedError


class ChromeTraceEventAdapter(EventAdapter):
    """Adapter for Chrome trace events in dictionary format (from JSON files)."""
    
    def get_time_range(self, event: dict[str, Any]) -> Optional[tuple[float, float]]:
        """Extract time range from dict event (microseconds)."""
        if event.get("ph") != "X":
            logger.debug(
                "Skipping event '%s': phase '%s' is not 'X' (Complete)",
                event.get("name", "unknown"),
                event.get("ph")
            )
            return None
        
        ts = event.get("ts")
        dur = event.get("dur")
        
        if ts is None:
            logger.debug(
                "Skipping event '%s': missing 'ts' field",
                event.get("name", "unknown")
            )
            return None
        if dur is None:
            logger.debug(
                "Skipping event '%s': missing 'dur' field",
                event.get("name", "unknown")
            )
            return None
        
        return (ts, ts + dur)
    
    def get_correlation_id(self, event: dict[str, Any]) -> Optional[int]:
        """Get correlation ID from dict event."""
        args = event.get("args", {})
        corr_id = args.get("correlation") or args.get("correlationId")
        if corr_id is None:
            logger.debug(
                "Event '%s' has no correlationId in args",
                event.get("name", "unknown")
            )
        return corr_id
    
    def get_event_id(self, event: dict[str, Any]) -> tuple:
        """Get unique identifier for dict event."""
        return (
            event.get("name", ""),
            event.get("ts"),
            event.get("pid"),
            event.get("tid")
        )


class NsysTraceEventAdapter(EventAdapter):
    """Adapter for ChromeTraceEvent Pydantic models parsed from nsys SQLite databases."""
    
    def get_time_range(self, event: ChromeTraceEvent) -> Optional[tuple[float, float]]:
        """Extract time range from ChromeTraceEvent (nanoseconds from args)."""
        if event.ph != "X":
            logger.debug(
                "Skipping event '%s': phase '%s' is not 'X' (Complete)",
                event.name,
                event.ph
            )
            return None
        
        start_ns = event.args.get("start_ns")
        end_ns = event.args.get("end_ns")
        
        if start_ns is None:
            logger.debug(
                "Skipping event '%s': missing 'start_ns' in args",
                event.name
            )
            return None
        if end_ns is None:
            logger.debug(
                "Skipping event '%s': missing 'end_ns' in args",
                event.name
            )
            return None
        
        return (start_ns, end_ns)
    
    def get_correlation_id(self, event: ChromeTraceEvent) -> Optional[int]:
        """Get correlation ID from ChromeTraceEvent."""
        corr_id = event.args.get("correlationId")
        if corr_id is None:
            logger.debug(
                "Event '%s' has no correlationId in args",
                event.name
            )
        return corr_id
    
    def get_event_id(self, event: ChromeTraceEvent) -> tuple:
        """Get unique identifier for ChromeTraceEvent."""
        return (
            event.name,
            event.args.get("start_ns"),
            event.args.get("deviceId"),
            event.args.get("raw_tid")
        )

