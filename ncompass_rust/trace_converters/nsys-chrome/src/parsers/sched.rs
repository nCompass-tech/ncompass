//! Thread scheduling event parser

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;

use crate::mapping::decompose_global_tid;
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

        let query = format!(
            "SELECT start, cpu, isSchedIn, globalTid, threadState, threadBlock FROM {}",
            self.table_name()
        );
        let mut stmt = context.conn.prepare(&query)?;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let start: i64 = row.get(0)?;
            let cpu: i32 = row.get(1)?;
            let is_sched_in: bool = row.get(2)?;
            let global_tid: i64 = row.get(3)?;
            let thread_state: Option<i32> = row.get(4)?;
            let thread_block: Option<i32> = row.get(5)?;

            let (pid, tid) = decompose_global_tid(global_tid);

            // Create instant event for scheduling change (like Python)
            let event_name = if is_sched_in {
                "Scheduled In"
            } else {
                "Scheduled Out"
            };

            // Use thread name lookup like Python, fallback to "Thread {tid}"
            let thread_name = context
                .thread_names
                .get(&tid)
                .cloned()
                .unwrap_or_else(|| format!("Thread {}", tid));

            let mut args = HashMap::default();
            args.insert("cpu".to_string(), json!(cpu));
            if let Some(ts) = thread_state {
                args.insert("threadState".to_string(), json!(ts));
            }
            if let Some(tb) = thread_block {
                args.insert("threadBlock".to_string(), json!(tb));
            }

            // Instant event (like Python uses ph="i")
            let mut event = ChromeTraceEvent::new(
                event_name.to_string(),
                crate::models::ChromeTracePhase::Instant,
                ns_to_us(start),
                format!("Process {}", pid),
                thread_name,
                "sched".to_string(),
            );
            event.args = args;

            events.push(event);
        }

        Ok(events)
    }
}

