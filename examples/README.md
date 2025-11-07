# nCompass SDK Examples

This directory contains example scripts demonstrating how to use the nCompass SDK for profiling and tracing AI inference workloads.

## Available Examples

The best way to use our tool is via our VSCode extension. However if you wish to use it fully programmatically, this directory contains a repository of examples.

### Basic Examples
- **[basic_example](basic_example/main.py)** - Basic example to get familiar with the concepts
- **[profiling_session](profiling_session/main.py)** - Complete ProfilingSession workflow

## Running Examples

### Prerequisites
* Ensure that you have a ncprof server running locally(not needed for `basic_example`)
* Ensure you create a .env file following the format of .example.env

Each example is self-contained and can be run directly:

### If running from your own workspace
```bash
pip install ncompass  # if not already installed
python examples/basic_example/main.py
```

### If running from this repo
```bash
nix develop
python examples/basic_example/main.py
```


## Support

For questions about examples:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- View the [API Reference](../docs/api_reference.md)

