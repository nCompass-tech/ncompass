#!/usr/bin/env python3
"""
Link user_annotation events to kernels in Chrome trace files.

This script takes a Chrome trace JSON file and creates new events that link
user_annotation events (like torch.profiler.record_function) to their
corresponding kernel executions using CUDA runtime correlation IDs.

Usage:
    python link_user_annotation_kernels.py input_trace.json output_trace.json
"""

import json
import sys
import argparse
from pathlib import Path

from ncompass.trace.converters import link_user_annotation_to_kernels
from utils import (
    count_events_by_category,
    print_event_statistics,
    calculate_replacement_sets,
    filter_replaced_events,
    print_replacement_statistics,
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Link user_annotation events to kernels in Chrome trace files"
    )
    parser.add_argument(
        "input_trace",
        type=str,
        help="Path to input Chrome trace JSON file"
    )
    parser.add_argument(
        "output_trace",
        type=str,
        help="Path to output Chrome trace JSON file"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_trace)
    output_path = Path(args.output_trace)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print("Linking user_annotation events to kernels...")
    new_events = link_user_annotation_to_kernels(str(input_path))
    
    if not new_events:
        print("No new events created. Trace may be missing required event types.")
        # Load trace to get original events for output
        with open(input_path, 'r') as f:
            trace_data = json.load(f)
        
        if isinstance(trace_data, list):
            linked_events = trace_data
        elif isinstance(trace_data, dict) and "traceEvents" in trace_data:
            linked_events = trace_data["traceEvents"]
        else:
            linked_events = []
    else:
        print(f"Created {len(new_events)} linked events")
        
        # Load trace to process results
        with open(input_path, 'r') as f:
            trace_data = json.load(f)
        
        # Handle both array and object formats
        if isinstance(trace_data, list):
            trace_events = trace_data
        elif isinstance(trace_data, dict) and "traceEvents" in trace_data:
            trace_events = trace_data["traceEvents"]
        else:
            trace_events = []
        
        # Count events for statistics
        counts = count_events_by_category(trace_events)
        print_event_statistics(counts)
        
        # Calculate replacement sets
        both_exist_names, ua_only_names = calculate_replacement_sets(new_events, trace_events)
        
        # Filter replaced events
        filtered_events, removed_gpu_ua_count, removed_ua_count = filter_replaced_events(
            trace_events, both_exist_names, ua_only_names
        )
        
        # Add new events
        linked_events = filtered_events + new_events
        
        # Print replacement statistics (always verbose)
        print_replacement_statistics(
            removed_gpu_ua_count,
            removed_ua_count,
            new_events,
            both_exist_names,
            ua_only_names,
            use_logger=False
        )
    
    print(f"\nWriting {len(linked_events)} events to {output_path}...")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(linked_events, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()

