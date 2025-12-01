"""nsys2chrome - Convert nsys SQLite exports to Chrome Trace JSON format."""

from .converter import NsysToChromeTraceConverter, convert_file, convert_nsys_report
from .models import ChromeTraceEvent, ConversionOptions
from .linker import link_user_annotation_to_kernels

__all__ = [
    "NsysToChromeTraceConverter",
    "convert_file",
    "convert_nsys_report",
    "ChromeTraceEvent",
    "ConversionOptions",
    "link_user_annotation_to_kernels",
]

__version__ = "0.1.0"

