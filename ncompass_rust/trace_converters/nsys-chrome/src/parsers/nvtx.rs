//! NVTX event parser

use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::collections::HashMap;

use crate::mapping::decompose_global_tid;
use crate::models::{ChromeTraceEvent, ns_to_us};
use crate::parsers::base::{EventParser, ParseContext};

/// NVTX Push/Pop event type ID (corresponds to torch.cuda.nvtx.range APIs)
const NVTX_PUSH_POP_EVENT_ID: i32 = 59;

/// Parser for NVTX_EVENTS table
pub struct NVTXParser;

impl NVTXParser {
    /// Build SQL WHERE clause for event prefix filtering
    fn build_filter_clause(event_prefix: &Option<Vec<String>>) -> String {
        match event_prefix {
            None => String::new(),
            Some(prefixes) if prefixes.is_empty() => String::new(),
            Some(prefixes) if prefixes.len() == 1 => {
                format!(" AND text LIKE '{}%'", prefixes[0])
            }
            Some(prefixes) => {
                let conditions: Vec<String> = prefixes
                    .iter()
                    .map(|prefix| format!("text LIKE '{}%'", prefix))
                    .collect();
                format!(" AND ({})", conditions.join(" OR "))
            }
        }
    }
}

impl EventParser for NVTXParser {
    fn table_name(&self) -> &str {
        "NVTX_EVENTS"
    }

    fn parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>> {
        let mut events = Vec::new();

        // Build compiled regex patterns for color scheme
        let color_patterns: Vec<(Regex, String)> = context
            .options
            .nvtx_color_scheme
            .iter()
            .filter_map(|(pattern, color)| {
                Regex::new(pattern)
                    .ok()
                    .map(|re| (re, color.clone()))
            })
            .collect();

        // Build filter clause for prefix filtering (done in SQL like Python)
        let filter_clause = Self::build_filter_clause(&context.options.nvtx_event_prefix);

        // Query with eventType filter (like Python) and optional prefix filter
        let query = format!(
            "SELECT start, end, text, textId, globalTid, eventType FROM {} WHERE eventType = {}{}",
            self.table_name(),
            NVTX_PUSH_POP_EVENT_ID,
            filter_clause
        );
        let mut stmt = context.conn.prepare(&query)?;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let start: i64 = row.get(0)?;
            let end: Option<i64> = row.get(1)?;
            let text: Option<String> = row.get(2)?;
            let text_id: Option<i32> = row.get(3)?;
            let global_tid: i64 = row.get(4)?;

            // Skip incomplete events (like Python)
            let end_time = match end {
                Some(e) => e,
                None => continue,
            };

            let (pid, tid) = decompose_global_tid(global_tid);
            let device_id = context.device_map.get(&pid).copied().unwrap_or(pid);

            // Resolve text: prefer textId lookup, fallback to text column, then "[No name]" (like Python)
            let event_name = if let Some(tid) = text_id {
                context
                    .strings
                    .get(&tid)
                    .cloned()
                    .unwrap_or_else(|| format!("[Unknown textId: {}]", tid))
            } else if let Some(ref t) = text {
                t.clone()
            } else {
                "[No name]".to_string()
            };

            let mut args = HashMap::default();
            args.insert("deviceId".to_string(), json!(device_id));
            args.insert("raw_pid".to_string(), json!(pid));
            args.insert("raw_tid".to_string(), json!(tid));
            args.insert("start_ns".to_string(), json!(start));
            args.insert("end_ns".to_string(), json!(end_time));

            let mut event = ChromeTraceEvent::complete(
                event_name.clone(),
                ns_to_us(start),
                ns_to_us(end_time - start),
                format!("Device {}", device_id),
                format!("NVTX Thread {}", tid),
                "nvtx".to_string(),
            )
            .with_args(args);

            // Apply color scheme if matches
            for (pattern, color) in &color_patterns {
                if pattern.is_match(&event_name) {
                    event = event.with_color(color.clone());
                    break;
                }
            }

            events.push(event);
        }

        Ok(events)
    }
}

