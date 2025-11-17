"""Command-line interface for nsys2chrome."""

import argparse
import json
from pathlib import Path

from .models import ConversionOptions
from .converter import convert_file


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Convert nsys SQLite export to Chrome Trace JSON format.'
    )
    parser.add_argument(
        "-f", "--filename",
        help="Path to the input SQLite file.",
        required=True
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file name, default to same as input with .json extension."
    )
    parser.add_argument(
        "-t", "--activity-type",
        help="Type of activities to include. Default to all available.",
        choices=['kernel', 'nvtx', 'nvtx-kernel', 'cuda-api', 'osrt', 'sched', 'composite'],
        nargs="+",
        default=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"]
    )
    parser.add_argument(
        "--nvtx-event-prefix",
        help="Filter NVTX events by their names' prefix.",
        type=str,
        nargs="*"
    )
    parser.add_argument(
        "--nvtx-color-scheme",
        help="""Color scheme for NVTX events.
        Accepts a dict mapping a string to one of chrome tracing colors.
        Events with names containing the string will be colored.
        E.g. '{"send": "thread_state_iowait", "recv": "thread_state_iowait", "compute": "thread_state_running"}'
        For details of the color scheme, see links in https://github.com/google/perfetto/issues/208
        """,
        type=json.loads,
        default={}
    )
    parser.add_argument(
        "--no-metadata",
        help="Don't include process/thread name metadata events.",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        args.output = Path(args.filename).with_suffix(".json")
    
    # Create conversion options
    options = ConversionOptions(
        activity_types=args.activity_type,
        nvtx_event_prefix=args.nvtx_event_prefix,
        nvtx_color_scheme=args.nvtx_color_scheme,
        include_metadata=not args.no_metadata,
    )
    
    # Perform conversion
    convert_file(args.filename, str(args.output), options)
    print(f"Conversion complete. Output written to {args.output}")


if __name__ == "__main__":
    main()

