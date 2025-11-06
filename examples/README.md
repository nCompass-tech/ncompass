# nCompass SDK Examples

This directory contains example scripts demonstrating how to use the nCompass SDK for profiling and tracing AI inference workloads.

## Available Examples

### Basic Examples

- **[basic_usage.py](basic_usage.py)** - Simple profiling session with trace analysis
- **[profiling_session.py](profiling_session.py)** - Complete ProfilingSession workflow
- **[advanced_tracing.py](advanced_tracing.py)** - Advanced features including iterative profiling

## Running Examples

### Prerequisites
Ensure that you have a ncprof server running locally

Each example is self-contained and can be run directly:

### If running from your own workspace
```bash
pip install ncompass  # if not already installed
python examples/basic_usage.py
```

### If running from this repo
```bash
nix develop
python examples/basic_usage.py
```


## Support

For questions about examples:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- View the [API Reference](../docs/api_reference.md)

