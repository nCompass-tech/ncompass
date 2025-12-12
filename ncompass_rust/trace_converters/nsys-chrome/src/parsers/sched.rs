//! Thread scheduling event parser

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;

use crate::models::{ChromeTraceEvent, ns_to_us};
use crate::parsers::base::{EventParser, ParseContext};

/// Parser for SCHED_EVENTS table
pub struct SchedParser;

impl EventParser for SchedParser {
    fn table_name(&self) -> &str {
        "SCHED_EVENTS"
    }

    fn parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>> {
        let mut events = Vec::new();

        let mut stmt = context.conn.prepare(&format!("SELECT * FROM {}", self.table_name()))?;
        let column_names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Find column indices - schema may vary
        let idx_start = column_names.iter().position(|n| n == "start" || n == "timestamp");
        let idx_end = column_names.iter().position(|n| n == "end");
        let idx_tid = column_names.iter().position(|n| n == "tid" || n == "threadId");
        let idx_name = column_names.iter().position(|n| n == "name" || n == "eventName");

        if idx_start.is_none() || idx_tid.is_none() {
            // Cannot parse without start and tid
            return Ok(events);
        }

        let idx_start = idx_start.unwrap();
        let idx_tid = idx_tid.unwrap();

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let start: i64 = row.get(idx_start)?;
            let tid: i32 = row.get(idx_tid)?;
            
            let end: Option<i64> = idx_end.and_then(|i| row.get(i).ok());
            let name: Option<String> = idx_name.and_then(|i| row.get(i).ok());

            let event_name = name.as_deref().unwrap_or("sched_event");

            // Use PID 0 for scheduling events (OS-level)
            let device_id = 0;

            let mut args = HashMap::default();
            args.insert("tid".to_string(), json!(tid));
            args.insert("start_ns".to_string(), json!(start));
            if let Some(end_time) = end {
                args.insert("end_ns".to_string(), json!(end_time));
            }

            let event = if let Some(end_time) = end {
                ChromeTraceEvent::complete(
                    event_name.to_string(),
                    ns_to_us(start),
                    ns_to_us(end_time - start),
                    format!("Device {}", device_id),
                    format!("Thread {}", tid),
                    "sched".to_string(),
                )
                .with_args(args)
            } else {
                // Instant event if no end time
                let mut event = ChromeTraceEvent::new(
                    event_name.to_string(),
                    crate::models::ChromeTracePhase::Instant,
                    ns_to_us(start),
                    format!("Device {}", device_id),
                    format!("Thread {}", tid),
                    "sched".to_string(),
                );
                event.args = args;
                event
            };

            events.push(event);
        }

        Ok(events)
    }
}

