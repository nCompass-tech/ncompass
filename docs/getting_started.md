# Getting Started with nCompass

This guide will help you get started with the nCompass Python SDK for AI inference profiling and tracing.

## Installation

Install the nCompass SDK using pip:

```bash
pip install ncompass
```

For development with PyTorch support:

```bash
pip install ncompass[torch]
```

## Quick Start

Here's a simple example to get you started with profiling:

```python
from ncompass.trace import ProfilingSession, enable_rewrites

# Initialize a profiling session
session = ProfilingSession(
    trace_output_dir="./traces",
    cache_dir="./cache"
)

# Define your model inference code
def run_inference():
    # Your inference code here
    pass

# Run profiling
trace_path = session.run_profile(run_inference)

# Get AI-powered trace analysis
summary = session.get_trace_summary(trace_path)
print(summary['markdown'])
```

## Basic Concepts

### ProfilingSession

The `ProfilingSession` is the main entry point for the SDK. It manages:
- Trace file collection and organization
- AI-powered performance analysis
- Iterative profiling workflows
- Configuration management

### Enable Rewrites

The `enable_rewrites()` function enables AST-level code instrumentation:

```python
from ncompass.trace import enable_rewrites, RewriteConfig

# Enable with configuration
config = RewriteConfig(targets={...})
enable_rewrites(config=config)
```

### Trace Analysis

Get detailed insights into your model's performance:

```python
# Get trace summary
summary = session.get_trace_summary()

# Access markdown report
print(summary['markdown'])

# Access structured data
structured_data = summary['structured']
```

## Next Steps
- Visit [Examples](../examples/) for complete working examples

## Requirements

- Python 3.8+
- PyTorch 2.0+ (optional, for torch profiling features)
- CUDA-capable GPU (optional, for GPU profiling)
- nCompass engine running locally