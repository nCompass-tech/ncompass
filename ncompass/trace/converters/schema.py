"""Schema detection and table discovery for nsys SQLite databases."""

import sqlite3
from typing import Any, Optional


def detect_available_tables(conn: sqlite3.Connection) -> set[str]:
    """Detect all available tables in the SQLite database.
    
    Args:
        conn: SQLite connection
        
    Returns:
        Set of table names
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    return tables


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    """Get column metadata for a table.
    
    Args:
        conn: SQLite connection
        table_name: Name of the table
        
    Returns:
        List of column info dictionaries with keys: name, type, notnull, default_value, pk
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = []
    for row in cursor.fetchall():
        columns.append({
            "name": row[1],
            "type": row[2],
            "notnull": bool(row[3]),
            "default_value": row[4],
            "pk": bool(row[5])
        })
    return columns


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database.
    
    Args:
        conn: SQLite connection
        table_name: Name of the table to check
        
    Returns:
        True if table exists, False otherwise
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


class TableRegistry:
    """Registry mapping table names to parser classes."""
    
    # Map table names to activity types
    TABLE_TO_ACTIVITY = {
        "CUPTI_ACTIVITY_KIND_KERNEL": "kernel",
        "CUPTI_ACTIVITY_KIND_RUNTIME": "cuda-api",
        "NVTX_EVENTS": "nvtx",
        "OSRT_API": "osrt",
        "SCHED_EVENTS": "sched",
        "COMPOSITE_EVENTS": "composite",
    }
    
    @classmethod
    def get_activity_type(cls, table_name: str) -> Optional[str]:
        """Get activity type for a table name.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Activity type string or None if not mapped
        """
        return cls.TABLE_TO_ACTIVITY.get(table_name)
    
    @classmethod
    def get_tables_for_activity(cls, activity_type: str) -> list[str]:
        """Get table names for an activity type.
        
        Args:
            activity_type: Activity type string
            
        Returns:
            List of table names that support this activity type
        """
        return [
            table for table, act_type in cls.TABLE_TO_ACTIVITY.items()
            if act_type == activity_type
        ]

