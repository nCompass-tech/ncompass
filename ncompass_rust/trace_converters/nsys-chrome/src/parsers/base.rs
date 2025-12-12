//! Base parser trait and shared parsing context

use anyhow::Result;
use rusqlite::Connection;
use std::collections::HashMap;

use crate::models::{ChromeTraceEvent, ConversionOptions};

/// Shared context for event parsing
pub struct ParseContext<'a> {
    /// SQLite connection
    pub conn: &'a Connection,
    /// String ID to string mapping
    pub strings: &'a HashMap<i32, String>,
    /// Conversion options
    pub options: &'a ConversionOptions,
    /// PID to device ID mapping
    pub device_map: &'a HashMap<i32, i32>,
    /// TID to thread name mapping
    pub thread_names: &'a HashMap<i32, String>,
}

impl<'a> ParseContext<'a> {
    pub fn new(
        conn: &'a Connection,
        strings: &'a HashMap<i32, String>,
        options: &'a ConversionOptions,
        device_map: &'a HashMap<i32, i32>,
        thread_names: &'a HashMap<i32, String>,
    ) -> Self {
        Self {
            conn,
            strings,
            options,
            device_map,
            thread_names,
        }
    }
}

/// Base trait for event parsers
pub trait EventParser {
    /// Get the table name this parser works with
    fn table_name(&self) -> &str;

    /// Parse events from the table
    fn parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>>;

    /// Safely parse events, returning empty list if table doesn't exist
    fn safe_parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>> {
        use crate::schema::table_exists;

        if !table_exists(context.conn, self.table_name())? {
            return Ok(Vec::new());
        }

        self.parse(context)
    }
}

