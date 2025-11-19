"""Default implementations for BaseParser methods."""

import sqlite3

from ..models import ChromeTraceEvent, ConversionOptions
from ..schema import table_exists as schema_table_exists
from ncompass.trace.infra.utils import logger


def default_init(parser, table_name: str):
    """Initialize parser.
    
    Args:
        parser: Parser instance
        table_name: Name of the SQLite table to parse
    """
    parser.table_name = table_name


def default_table_exists(parser, conn: sqlite3.Connection) -> bool:
    """Check if the table exists in the database.
    
    Args:
        parser: Parser instance
        conn: SQLite connection
        
    Returns:
        True if table exists, False otherwise
    """
    return schema_table_exists(conn, parser.table_name)


def default_safe_parse(
    parser,
    conn: sqlite3.Connection,
    strings: dict[int, str],
    options: ConversionOptions,
    device_map: dict[int, int],
    thread_names: dict[int, str],
) -> list[ChromeTraceEvent]:
    """Safely parse events, returning empty list if table doesn't exist.
    
    Args:
        parser: Parser instance
        conn: SQLite connection
        strings: String ID to string mapping
        options: Conversion options
        device_map: PID to device ID mapping
        thread_names: TID to thread name mapping
        
    Returns:
        List of Chrome Trace events, or empty list if table doesn't exist
    """
    if not parser.table_exists(conn):
        return []
    
    try:
        return parser.parse(conn, strings, options, device_map, thread_names)
    except Exception as e:
        logger.warning(
            f"Failed to parse {parser.table_name}: {e}",
        )
        return []
