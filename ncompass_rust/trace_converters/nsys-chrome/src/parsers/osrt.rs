//! OS Runtime API event parser

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;

use crate::mapping::decompose_global_tid;
use crate::models::{ChromeTraceEvent, ns_to_us};
use crate::parsers::base::{EventParser, ParseContext};

/// Parser for OSRT_API table
pub struct OSRTParser;

impl EventParser for OSRTParser {
    fn table_name(&self) -> &str {
        "OSRT_API"
    }

    fn parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>> {
        let mut events = Vec::new();

        let mut stmt = context.conn.prepare(&format!("SELECT * FROM {}", self.table_name()))?;
        let column_names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Find column indices
        let idx_start = column_names.iter().position(|n| n == "start").unwrap();
        let idx_end = column_names.iter().position(|n| n == "end").unwrap();
        let idx_global_tid = column_names.iter().position(|n| n == "globalTid").unwrap();
        let idx_name_id = column_names.iter().position(|n| n == "nameId").unwrap();

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let start: i64 = row.get(idx_start)?;
            let end: i64 = row.get(idx_end)?;
            let global_tid: i64 = row.get(idx_global_tid)?;
            let name_id: i32 = row.get(idx_name_id)?;

            let (pid, tid) = decompose_global_tid(global_tid);

            let api_name = context
                .strings
                .get(&name_id)
                .map(|s| s.as_str())
                .unwrap_or("Unknown OSRT API");

            // Use thread name lookup like Python, fallback to "Thread {tid}"
            let thread_name = context
                .thread_names
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("Thread {}", tid));

            let mut args = HashMap::default();
            args.insert("raw_pid".to_string(), json!(pid));
            args.insert("raw_tid".to_string(), json!(tid));
            args.insert("start_ns".to_string(), json!(start));
            args.insert("end_ns".to_string(), json!(end));

            let event = ChromeTraceEvent::complete(
                api_name.to_string(),
                ns_to_us(start),
                ns_to_us(end - start),
                format!("Process {}", pid),
                thread_name,
                "osrt".to_string(),
            )
            .with_args(args);

            events.push(event);
        }

        Ok(events)
    }
}

