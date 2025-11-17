"""Base parser class for nsys event parsers."""

import sqlite3
from abc import ABC, abstractmethod
from typing import Any

from ..models import ChromeTraceEvent, ConversionOptions
from ..schema import table_exists


class BaseParser(ABC):
    """Abstract base class for nsys event parsers."""
    
    def __init__(self, table_name: str):
        """Initialize parser.
        
        Args:
            table_name: Name of the SQLite table to parse
        """
        self.table_name = table_name
    
    def table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check if the table exists in the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            True if table exists, False otherwise
        """
        return table_exists(conn, self.table_name)
    
    @abstractmethod
    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Parse events from the table.
        
        Args:
            conn: SQLite connection
            strings: String ID to string mapping
            options: Conversion options
            device_map: PID to device ID mapping
            thread_names: TID to thread name mapping
            
        Returns:
            List of Chrome Trace events
        """
        pass
    
    def safe_parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> list[ChromeTraceEvent]:
        """Safely parse events, returning empty list if table doesn't exist.
        
        Args:
            conn: SQLite connection
            strings: String ID to string mapping
            options: Conversion options
            device_map: PID to device ID mapping
            thread_names: TID to thread name mapping
            
        Returns:
            List of Chrome Trace events, or empty list if table doesn't exist
        """
        if not self.table_exists(conn):
            return []
        
        try:
            return self.parse(conn, strings, options, device_map, thread_names)
        except Exception as e:
            # Log warning but don't fail completely
            import warnings
            warnings.warn(
                f"Failed to parse {self.table_name}: {e}",
                UserWarning
            )
            return []

