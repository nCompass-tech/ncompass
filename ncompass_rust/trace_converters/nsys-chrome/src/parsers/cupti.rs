//! CUPTI event parsers for CUDA kernel and runtime events

use anyhow::Result;
use serde_json::json;
use std::collections::HashMap;

use crate::mapping::decompose_global_tid;
use crate::models::{ChromeTraceEvent, ns_to_us};
use crate::parsers::base::{EventParser, ParseContext};

/// Parser for CUPTI_ACTIVITY_KIND_KERNEL table
pub struct CUPTIKernelParser;

impl EventParser for CUPTIKernelParser {
    fn table_name(&self) -> &str {
        "CUPTI_ACTIVITY_KIND_KERNEL"
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
        let idx_device = column_names.iter().position(|n| n == "deviceId").unwrap();
        let idx_stream = column_names.iter().position(|n| n == "streamId").unwrap();
        let idx_short_name = column_names.iter().position(|n| n == "shortName").unwrap();
        let idx_start = column_names.iter().position(|n| n == "start").unwrap();
        let idx_end = column_names.iter().position(|n| n == "end").unwrap();
        let idx_grid_x = column_names.iter().position(|n| n == "gridX").unwrap();
        let idx_grid_y = column_names.iter().position(|n| n == "gridY").unwrap();
        let idx_grid_z = column_names.iter().position(|n| n == "gridZ").unwrap();
        let idx_block_x = column_names.iter().position(|n| n == "blockX").unwrap();
        let idx_block_y = column_names.iter().position(|n| n == "blockY").unwrap();
        let idx_block_z = column_names.iter().position(|n| n == "blockZ").unwrap();
        let idx_regs = column_names.iter().position(|n| n == "registersPerThread").unwrap();
        let idx_static_smem = column_names.iter().position(|n| n == "staticSharedMemory").unwrap();
        let idx_dynamic_smem = column_names.iter().position(|n| n == "dynamicSharedMemory").unwrap();
        let idx_corr = column_names.iter().position(|n| n == "correlationId").unwrap();

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let device_id: i32 = row.get(idx_device)?;
            let stream_id: i32 = row.get(idx_stream)?;
            let short_name_id: i32 = row.get(idx_short_name)?;
            let start: i64 = row.get(idx_start)?;
            let end: i64 = row.get(idx_end)?;
            let grid_x: i32 = row.get(idx_grid_x)?;
            let grid_y: i32 = row.get(idx_grid_y)?;
            let grid_z: i32 = row.get(idx_grid_z)?;
            let block_x: i32 = row.get(idx_block_x)?;
            let block_y: i32 = row.get(idx_block_y)?;
            let block_z: i32 = row.get(idx_block_z)?;
            let regs_per_thread: i32 = row.get(idx_regs)?;
            let static_smem: i32 = row.get(idx_static_smem)?;
            let dynamic_smem: i32 = row.get(idx_dynamic_smem)?;
            let correlation_id: i32 = row.get(idx_corr)?;

            let kernel_name = context
                .strings
                .get(&short_name_id)
                .map(|s| s.as_str())
                .unwrap_or("Unknown Kernel");

            let mut args = HashMap::default();
            args.insert("grid".to_string(), json!([grid_x, grid_y, grid_z]));
            args.insert("block".to_string(), json!([block_x, block_y, block_z]));
            args.insert("registersPerThread".to_string(), json!(regs_per_thread));
            args.insert("staticSharedMemory".to_string(), json!(static_smem));
            args.insert("dynamicSharedMemory".to_string(), json!(dynamic_smem));
            args.insert("correlationId".to_string(), json!(correlation_id));
            args.insert("deviceId".to_string(), json!(device_id));
            args.insert("streamId".to_string(), json!(stream_id));
            args.insert("start_ns".to_string(), json!(start));
            args.insert("end_ns".to_string(), json!(end));

            let event = ChromeTraceEvent::complete(
                kernel_name.to_string(),
                ns_to_us(start),
                ns_to_us(end - start),
                format!("Device {}", device_id),
                format!("Stream {}", stream_id),
                "kernel".to_string(),
            )
            .with_args(args);

            events.push(event);
        }

        Ok(events)
    }
}

/// Parser for CUPTI_ACTIVITY_KIND_RUNTIME table
pub struct CUPTIRuntimeParser;

impl EventParser for CUPTIRuntimeParser {
    fn table_name(&self) -> &str {
        "CUPTI_ACTIVITY_KIND_RUNTIME"
    }

    fn parse(&self, context: &ParseContext) -> Result<Vec<ChromeTraceEvent>> {
        let mut events = Vec::new();

        let query = format!(
            "SELECT start, end, globalTid, correlationId, nameId FROM {}",
            self.table_name()
        );
        let mut stmt = context.conn.prepare(&query)?;

        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let start: i64 = row.get(0)?;
            let end: i64 = row.get(1)?;
            let global_tid: i64 = row.get(2)?;
            let correlation_id: i32 = row.get(3)?;
            let name_id: i32 = row.get(4)?;

            let (pid, tid) = decompose_global_tid(global_tid);
            let device_id = context.device_map.get(&pid).copied().unwrap_or(pid);

            let api_name = context
                .strings
                .get(&name_id)
                .map(|s| s.as_str())
                .unwrap_or("Unknown API");

            let mut args = HashMap::default();
            args.insert("correlationId".to_string(), json!(correlation_id));
            args.insert("deviceId".to_string(), json!(device_id));
            args.insert("raw_pid".to_string(), json!(pid));
            args.insert("raw_tid".to_string(), json!(tid));
            args.insert("start_ns".to_string(), json!(start));
            args.insert("end_ns".to_string(), json!(end));

            let event = ChromeTraceEvent::complete(
                api_name.to_string(),
                ns_to_us(start),
                ns_to_us(end - start),
                format!("Device {}", device_id),
                format!("CUDA API Thread {}", tid),
                "cuda_api".to_string(),
            )
            .with_args(args);

            events.push(event);
        }

        Ok(events)
    }
}

