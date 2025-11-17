# nsys2chrome

A library for converting nsys SQLite exports to Chrome Trace JSON format (Perfetto-compatible).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### As a Library

```python
from nsys2chrome import NsysToChromeTraceConverter, ConversionOptions

# Create conversion options
options = ConversionOptions(
    activity_types=["kernel", "nvtx", "osrt"],
    nvtx_color_scheme={"compute": "thread_state_running"}
)

# Convert using context manager
with NsysToChromeTraceConverter("input.sqlite", options) as converter:
    events = converter.convert()
    # events is a list of ChromeTraceEvent objects

# Or use the convenience function
from nsys2chrome import convert_file
convert_file("input.sqlite", "output.json", options)
```

### Command Line Interface

```bash
python -m nsys2chrome.cli -f input.sqlite -o output.json -t kernel nvtx osrt
```

## Features

- **Robust**: Handles missing tables gracefully
- **Type-safe**: Uses Pydantic models for validation
- **Modular**: Extensible parser architecture
- **Multi-GPU support**: Handles multiple devices
- **Multi-thread support**: Proper thread/process mapping

## Supported Event Types

- `kernel` - CUDA kernel events
- `cuda-api` - CUDA runtime API events
- `nvtx` - NVTX annotation events
- `nvtx-kernel` - NVTX events linked to kernel execution times
- `osrt` - OS runtime API events
- `sched` - Thread scheduling events
- `composite` - Composite events

## Requirements

- Python 3.8+
- pydantic >= 2.0.0

