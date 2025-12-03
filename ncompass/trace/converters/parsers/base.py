"""Base parser class for nsys event parsers."""

import sqlite3
from abc import ABC, abstractmethod
from typing import Any, Iterator

from ..models import ChromeTraceEvent, ConversionOptions
from ..schema import table_exists
from ncompass.types import Trait

from ncompass.trace.infra.utils import logger

class BaseParser(Trait):
    """Abstract base class for nsys event parsers.
    
    All parsers use generators for memory-efficient streaming.
    """
    
    def __init__(self, table_name: str):
        """Initialize parser.
        
        Args:
            table_name: Name of the SQLite table to parse
        """
        raise NotImplementedError
    
    def table_exists(self, conn: sqlite3.Connection) -> bool:
        """Check if the table exists in the database.
        
        Args:
            conn: SQLite connection
            
        Returns:
            True if table exists, False otherwise
        """
        raise NotImplementedError
    
    def parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> Iterator[ChromeTraceEvent]:
        """Parse events from the table (streaming).
        
        Args:
            conn: SQLite connection
            strings: String ID to string mapping
            options: Conversion options
            device_map: PID to device ID mapping
            thread_names: TID to thread name mapping
            
        Yields:
            Chrome Trace events one at a time
        """
        raise NotImplementedError
    
    def safe_parse(
        self,
        conn: sqlite3.Connection,
        strings: dict[int, str],
        options: ConversionOptions,
        device_map: dict[int, int],
        thread_names: dict[int, str],
    ) -> Iterator[ChromeTraceEvent]:
        """Safely parse events, yielding nothing if table doesn't exist.
        
        Args:
            conn: SQLite connection
            strings: String ID to string mapping
            options: Conversion options
            device_map: PID to device ID mapping
            thread_names: TID to thread name mapping
            
        Yields:
            Chrome Trace events one at a time, or nothing if table doesn't exist
        """
        raise NotImplementedError
