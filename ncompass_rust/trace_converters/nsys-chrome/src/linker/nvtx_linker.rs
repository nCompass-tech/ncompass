//! Link NVTX events to kernel events via CUDA API correlation

use log::debug;
use regex::Regex;
use std::collections::{HashMap, HashSet};

use crate::linker::adapters::{EventAdapter, NsysEventAdapter};
use crate::linker::algorithms::{
    aggregate_kernel_times, build_correlation_map, find_kernels_for_annotation,
    find_overlapping_intervals,
};
use crate::models::{BindingPoint, ChromeTraceEvent, ConversionOptions, StringOrInt, ns_to_us};

/// Link NVTX events to kernel events via CUDA API correlation
pub fn link_nvtx_to_kernels<'a>(
    nvtx_events: &'a [ChromeTraceEvent],
    cuda_api_events: &'a [ChromeTraceEvent],
    kernel_events: &'a [ChromeTraceEvent],
    options: &ConversionOptions,
) -> (
    Vec<ChromeTraceEvent>,
    HashSet<(i32, i32, i64, String)>,
    Vec<ChromeTraceEvent>,
) {
    // Group events by device ID
    let (per_device_nvtx, per_device_cuda_api, per_device_kernels) =
        group_events_by_device(nvtx_events, cuda_api_events, kernel_events);

    // Get devices that have all three event types
    let common_devices: HashSet<i32> = per_device_nvtx
        .keys()
        .copied()
        .collect::<HashSet<_>>()
        .intersection(&per_device_cuda_api.keys().copied().collect())
        .copied()
        .collect::<HashSet<_>>()
        .intersection(&per_device_kernels.keys().copied().collect())
        .copied()
        .collect();

    // Create adapter
    let adapter = NsysEventAdapter;

    // Process each device
    let mut all_nvtx_kernel_events = Vec::new();
    let mut all_mapped_nvtx_identifiers = HashSet::new();
    let mut all_flow_events = Vec::new();

    for &device_id in &common_devices {
        let (nvtx_kernel_events, mapped_nvtx_identifiers, flow_events) = process_device_nvtx_events(
            &per_device_nvtx[&device_id],
            &per_device_cuda_api[&device_id],
            &per_device_kernels[&device_id],
            device_id,
            &adapter,
            options,
        );

        all_nvtx_kernel_events.extend(nvtx_kernel_events);
        all_mapped_nvtx_identifiers.extend(mapped_nvtx_identifiers);
        all_flow_events.extend(flow_events);
    }

    (
        all_nvtx_kernel_events,
        all_mapped_nvtx_identifiers,
        all_flow_events,
    )
}

/// Group events by device ID
pub(crate) fn group_events_by_device<'a>(
    nvtx_events: &'a [ChromeTraceEvent],
    cuda_api_events: &'a [ChromeTraceEvent],
    kernel_events: &'a [ChromeTraceEvent],
) -> (
    HashMap<i32, Vec<&'a ChromeTraceEvent>>,
    HashMap<i32, Vec<&'a ChromeTraceEvent>>,
    HashMap<i32, Vec<&'a ChromeTraceEvent>>,
) {
    let mut per_device_nvtx: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::default();
    let mut per_device_cuda_api: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::default();
    let mut per_device_kernels: HashMap<i32, Vec<&ChromeTraceEvent>> = HashMap::default();

    let mut nvtx_no_device = 0;
    let mut nvtx_no_times = 0;
    for event in nvtx_events {
        if let Some(device_id) = event.args.get("deviceId").and_then(|v| v.as_i64()) {
            let has_times = event.args.get("start_ns").is_some() && event.args.get("end_ns").is_some();
            if has_times {
                per_device_nvtx
                    .entry(device_id as i32)
                    .or_insert_with(Vec::new)
                    .push(event);
            } else {
                nvtx_no_times += 1;
            }
        } else {
            nvtx_no_device += 1;
        }
    }

    let mut cuda_api_no_device = 0;
    let mut cuda_api_no_corr = 0;
    for event in cuda_api_events {
        if let Some(device_id) = event.args.get("deviceId").and_then(|v| v.as_i64()) {
            if event.args.get("correlationId").is_some() {
                per_device_cuda_api
                    .entry(device_id as i32)
                    .or_insert_with(Vec::new)
                    .push(event);
            } else {
                cuda_api_no_corr += 1;
            }
        } else {
            cuda_api_no_device += 1;
        }
    }

    let mut kernel_no_device = 0;
    let mut kernel_no_corr = 0;
    for event in kernel_events {
        if let Some(device_id) = event.args.get("deviceId").and_then(|v| v.as_i64()) {
            if event.args.get("correlationId").is_some() {
                per_device_kernels
                    .entry(device_id as i32)
                    .or_insert_with(Vec::new)
                    .push(event);
            } else {
                kernel_no_corr += 1;
            }
        } else {
            kernel_no_device += 1;
        }
    }

    // Log summary of filtered events
    if nvtx_no_device > 0 || nvtx_no_times > 0 {
        debug!(
            "group_events_by_device: filtered {} NVTX events (no deviceId: {}, no times: {})",
            nvtx_no_device + nvtx_no_times,
            nvtx_no_device,
            nvtx_no_times
        );
    }
    if cuda_api_no_device > 0 || cuda_api_no_corr > 0 {
        debug!(
            "group_events_by_device: filtered {} CUDA API events (no deviceId: {}, no correlationId: {})",
            cuda_api_no_device + cuda_api_no_corr,
            cuda_api_no_device,
            cuda_api_no_corr
        );
    }
    if kernel_no_device > 0 || kernel_no_corr > 0 {
        debug!(
            "group_events_by_device: filtered {} kernel events (no deviceId: {}, no correlationId: {})",
            kernel_no_device + kernel_no_corr,
            kernel_no_device,
            kernel_no_corr
        );
    }

    (per_device_nvtx, per_device_cuda_api, per_device_kernels)
}

/// Process NVTX events for a single device
fn process_device_nvtx_events(
    nvtx_events_list: &[&ChromeTraceEvent],
    cuda_api_events_list: &[&ChromeTraceEvent],
    kernel_events_list: &[&ChromeTraceEvent],
    device_id: i32,
    adapter: &NsysEventAdapter,
    options: &ConversionOptions,
) -> (
    Vec<ChromeTraceEvent>,
    HashSet<(i32, i32, i64, String)>,
    Vec<ChromeTraceEvent>,
) {
    let mut nvtx_kernel_events = Vec::new();
    let mut mapped_nvtx_identifiers = HashSet::new();

    // Find overlapping intervals between NVTX and CUDA API events
    let overlap_map = find_overlapping_intervals(nvtx_events_list, cuda_api_events_list, adapter);

    // Build correlation ID map
    let correlation_id_map = build_correlation_map_with_cuda_api(cuda_api_events_list, kernel_events_list, adapter);

    // Generate flow events
    let flow_events = generate_flow_events_for_correlation_map(&correlation_id_map);

    // Extract kernel correlation map for finding kernels
    let kernel_correlation_map: HashMap<i32, Vec<&ChromeTraceEvent>> = correlation_id_map
        .iter()
        .map(|(&corr_id, data)| (corr_id, data.kernels.clone()))
        .collect();

    // Process each NVTX event
    for nvtx_event in nvtx_events_list {
        let nvtx_id = adapter.get_event_id(nvtx_event);
        let cuda_api_events_overlapping = overlap_map.get(&nvtx_id).map(|v| v.as_slice()).unwrap_or(&[]);

        if cuda_api_events_overlapping.is_empty() {
            continue;
        }

        // Find kernels using shared function
        let found_kernels = find_kernels_for_annotation(
            cuda_api_events_overlapping,
            &kernel_correlation_map,
            adapter,
        );

        // Aggregate kernel times
        if let Some((kernel_start_time, kernel_end_time)) =
            aggregate_kernel_times(&found_kernels, adapter)
        {
            // Create nvtx-kernel event
            let event = create_nvtx_kernel_event(
                nvtx_event,
                kernel_start_time,
                kernel_end_time,
                device_id,
                options,
            );
            nvtx_kernel_events.push(event);

            // Track this NVTX event as successfully mapped
            if let (Some(tid), Some(start_ns)) = (
                nvtx_event.args.get("raw_tid").and_then(|v| v.as_i64()),
                nvtx_event.args.get("start_ns").and_then(|v| v.as_i64()),
            ) {
                let nvtx_identifier = (device_id, tid as i32, start_ns, nvtx_event.name.clone());
                mapped_nvtx_identifiers.insert(nvtx_identifier);
            }
        }
    }

    (nvtx_kernel_events, mapped_nvtx_identifiers, flow_events)
}

/// Correlation data for CUDA API and kernels
struct CorrelationData<'a> {
    cuda_api: Option<&'a ChromeTraceEvent>,
    kernels: Vec<&'a ChromeTraceEvent>,
}

/// Build correlation ID map including both CUDA API and kernel events
fn build_correlation_map_with_cuda_api<'a>(
    cuda_api_events_list: &[&'a ChromeTraceEvent],
    kernel_events_list: &[&'a ChromeTraceEvent],
    adapter: &NsysEventAdapter,
) -> HashMap<i32, CorrelationData<'a>> {
    let mut correlation_id_map: HashMap<i32, CorrelationData> = HashMap::default();

    // Map CUDA API events by correlationId
    for &cuda_api_event in cuda_api_events_list {
        if let Some(corr_id) = adapter.get_correlation_id(cuda_api_event) {
            correlation_id_map
                .entry(corr_id)
                .or_insert_with(|| CorrelationData {
                    cuda_api: None,
                    kernels: Vec::new(),
                })
                .cuda_api = Some(cuda_api_event);
        }
    }

    // Map kernel events by correlationId
    let kernel_correlation_map = build_correlation_map(kernel_events_list, adapter);
    for (corr_id, kernels) in kernel_correlation_map {
        correlation_id_map
            .entry(corr_id)
            .or_insert_with(|| CorrelationData {
                cuda_api: None,
                kernels: Vec::new(),
            })
            .kernels = kernels;
    }

    correlation_id_map
}

/// Generate flow events for all CUDA API → Kernel links
fn generate_flow_events_for_correlation_map(
    correlation_id_map: &HashMap<i32, CorrelationData>,
) -> Vec<ChromeTraceEvent> {
    let mut flow_events = Vec::new();

    for (&corr_id, data) in correlation_id_map {
        if let Some(cuda_api_event) = data.cuda_api {
            if !data.kernels.is_empty() {
                // Create flow arrow to EACH kernel
                for &kernel_event in &data.kernels {
                    let (flow_start, flow_finish) =
                        create_flow_events(cuda_api_event, kernel_event, corr_id);
                    flow_events.push(flow_start);
                    flow_events.push(flow_finish);
                }
            }
        }
    }

    flow_events
}

/// Create flow start/end events to show arrows in Perfetto
pub(crate) fn create_flow_events(
    cuda_api_event: &ChromeTraceEvent,
    kernel_event: &ChromeTraceEvent,
    correlation_id: i32,
) -> (ChromeTraceEvent, ChromeTraceEvent) {
    let flow_start = ChromeTraceEvent::flow_start(
        cuda_api_event.ts,
        cuda_api_event.pid.clone(),
        cuda_api_event.tid.clone(),
        StringOrInt::Int(correlation_id as i64),
    );

    let flow_finish = ChromeTraceEvent::flow_finish(
        kernel_event.ts,
        kernel_event.pid.clone(),
        kernel_event.tid.clone(),
        StringOrInt::Int(correlation_id as i64),
        BindingPoint::Enclosing,
    );

    (flow_start, flow_finish)
}

/// Create a single nvtx-kernel event from an NVTX event and kernel time range
pub(crate) fn create_nvtx_kernel_event(
    nvtx_event: &ChromeTraceEvent,
    kernel_start_time: i64,
    kernel_end_time: i64,
    device_id: i32,
    options: &ConversionOptions,
) -> ChromeTraceEvent {
    let nvtx_name = &nvtx_event.name;
    let tid = nvtx_event
        .args
        .get("raw_tid")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let mut event = ChromeTraceEvent::complete(
        nvtx_name.clone(),
        ns_to_us(kernel_start_time),
        ns_to_us(kernel_end_time - kernel_start_time),
        format!("Device {}", device_id),
        format!("NVTX Kernel Thread {}", tid),
        "nvtx-kernel".to_string(),
    );

    // Apply color scheme if specified
    for (pattern_str, color) in &options.nvtx_color_scheme {
        if let Ok(pattern) = Regex::new(pattern_str) {
            if pattern.is_match(nvtx_name) {
                event = event.with_color(color.clone());
                break;
            }
        }
    }

    event
}

