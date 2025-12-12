//! NVTX event parser

use anyhow::Result;
use regex::Regex;
use serde_json::json;
use std::collections::HashMap;

use crate::mapping::decompose_global_tid;
use crate::models::{ChromeTraceEvent, ns_to_us};
use crate::parsers::base::{EventParser, ParseContext};

/// Parser for NVTX_EVENTS table
pub struct NVTXParser;

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

        let mut stmt = context.conn.prepare(&format!("SELECT * FROM {}", self.table_name()))?;
        let column_names: Vec<String> = stmt
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Find column indices
        let idx_text = column_names.iter().position(|n| n == "text");
        let idx_start = column_names.iter().position(|n| n == "start").unwrap();
        let idx_end = column_names.iter().position(|n| n == "end");
        let idx_global_tid = column_names.iter().position(|n| n == "globalTid").unwrap();

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let text: Option<String> = idx_text.and_then(|i| row.get(i).ok());
            let start: i64 = row.get(idx_start)?;
            let end: Option<i64> = idx_end.and_then(|i| row.get(i).ok());
            let global_tid: i64 = row.get(idx_global_tid)?;

            let event_name = text.as_deref().unwrap_or("NVTX Range");

            // Apply prefix filtering if specified
            if let Some(ref prefixes) = context.options.nvtx_event_prefix {
                let matches_prefix = prefixes.iter().any(|prefix| event_name.starts_with(prefix));
                if !matches_prefix {
                    continue;
                }
            }

            let (pid, tid) = decompose_global_tid(global_tid);
            let device_id = context.device_map.get(&pid).copied().unwrap_or(pid);

            // Only create events with both start and end times
            if let Some(end_time) = end {
                let mut args = HashMap::default();
                args.insert("deviceId".to_string(), json!(device_id));
                args.insert("raw_tid".to_string(), json!(tid));
                args.insert("start_ns".to_string(), json!(start));
                args.insert("end_ns".to_string(), json!(end_time));

                let mut event = ChromeTraceEvent::complete(
                    event_name.to_string(),
                    ns_to_us(start),
                    ns_to_us(end_time - start),
                    format!("Device {}", device_id),
                    format!("NVTX Thread {}", tid),
                    "nvtx".to_string(),
                )
                .with_args(args);

                // Apply color scheme if matches
                for (pattern, color) in &color_patterns {
                    if pattern.is_match(event_name) {
                        event = event.with_color(color.clone());
                        break;
                    }
                }

                events.push(event);
            }
        }

        Ok(events)
    }
}

