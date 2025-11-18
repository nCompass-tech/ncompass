# nsys Report Conversion with nCompass SDK

This example demonstrates how to convert nsys profiling reports (`.nsys-rep` files) to Chrome trace JSON format using the nCompass SDK. The conversion process involves two steps:

1. **nsys-rep → SQLite**: Convert the binary nsys report to SQLite format using the `nsys` CLI tool
2. **SQLite → JSON**: Convert the SQLite database to Chrome trace JSON format using the nCompass SDK

## Overview

This example shows how to:
- Convert existing nsys reports to SQLite format
- Use the nCompass SDK to convert SQLite databases to Chrome trace JSON format
- Visualize GPU profiling data in Perfetto or Chrome DevTools

## Prerequisites

- **nsys CLI tool**: The NVIDIA Nsight Systems CLI must be installed and available in your PATH
  - Download from: [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
  - Verify installation: `nsys --version`
- **Python 3.9+**: Required for running the conversion script
- **nCompass SDK**: Will be installed via requirements.txt

## Installation

### Option 1: Local Installation (venv)

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

For users who prefer to run in a Docker container with Nsight Systems pre-installed:

1. Build the Docker image:
```bash
python nc_pkg.py --build
```

2. Run the Docker container:
```bash
python nc_pkg.py --run
```

This will build and run a Docker container with:
- NVIDIA CUDA toolkit
- Nsight Systems CLI (nsys) pre-installed
- nCompass SDK and dependencies
- All necessary tools for conversion

The container mounts the current directory to `/workspace`, so you can access your files and run the conversion script inside the container.

**Note**: Docker installation requires Docker and NVIDIA Container Toolkit to be installed on your system.

## Usage

### Basic Conversion

Convert the default `vllm_trace_profile.nsys-rep` file to both SQLite and JSON formats:

```bash
python convert_nsys.py
```

This will generate:
- `vllm_trace_profile.sqlite` - SQLite database export
- `vllm_trace_profile.json` - Chrome trace JSON file

### Convert a Specific File

Convert a custom nsys report file:

```bash
python convert_nsys.py --input my_profile.nsys-rep
```

### Custom Output Name

Specify a custom base name for output files:

```bash
python convert_nsys.py --input my_profile.nsys-rep --output my_trace
```

This generates `my_trace.sqlite` and `my_trace.json`.

### Run Individual Steps

Run only the SQLite conversion step:

```bash
python convert_nsys.py --step sqlite --input my_profile.nsys-rep
```

Run only the Chrome trace conversion step (requires an existing SQLite file):

```bash
python convert_nsys.py --step chrome --input my_profile.sqlite
```

## Command-Line Options

- `--input`, `-i`: Input nsys report file (`.nsys-rep`) or SQLite file (`.sqlite`) if using `--step chrome`. Default: `vllm_trace_profile.nsys-rep`
- `--output`, `-o`: Base name for output files (without extensions). If not specified, uses input filename without extension.
- `--step`, `-s`: Which step to run: `sqlite` (nsys-rep → sqlite), `chrome` (sqlite → json), or `all` (run all steps). Default: `all`

## Output Files

### SQLite File (`.sqlite`)

The SQLite database contains structured profiling data exported from the nsys report. This intermediate format allows for programmatic querying and analysis of profiling data.

### Chrome Trace JSON (`.json`)

The Chrome trace JSON file is compatible with:
- **Perfetto UI**: [ui.perfetto.dev](https://ui.perfetto.dev) - Upload and visualize the trace file
- **Chrome DevTools**: Open Chrome DevTools → Performance tab → Load profile
- **nCompass VSCode Extension**: View traces directly in VSCode

The trace includes:
- **CUDA kernels**: GPU kernel execution events
- **NVTX markers**: User-defined annotations and ranges
- **CUDA API calls**: Runtime API events
- **OS runtime events**: System-level events
- **Thread scheduling**: CPU thread scheduling information

## How It Works

### Step 1: nsys-rep → SQLite

The script uses the `nsys export` command to convert the binary nsys report to SQLite format:

```bash
nsys export --type sqlite --include-json true --force-overwrite true -o output.sqlite input.nsys-rep
```

### Step 2: SQLite → Chrome Trace JSON

The nCompass SDK's `convert_file` function reads the SQLite database and converts it to Chrome trace format:

```python
from ncompass.trace.converters import convert_file, ConversionOptions

options = ConversionOptions(
    activity_types=["kernel", "nvtx", "nvtx-kernel", "cuda-api", "osrt", "sched"],
    include_metadata=True
)
convert_file("input.sqlite", "output.json", options)
```

The conversion process:
1. Reads event data from SQLite tables
2. Maps events to Chrome trace format
3. Links NVTX markers to kernel execution times
4. Adds metadata events for process/thread names
5. Writes the final JSON trace file

## Viewing Traces

### Using Perfetto UI

1. Go to [ui.perfetto.dev](https://ui.perfetto.dev)
2. Click "Open trace file"
3. Select your `.json` file
4. Explore the timeline view with GPU kernels, NVTX markers, and CPU events

### Using Chrome DevTools

1. Open Chrome and press F12 to open DevTools
2. Go to the "Performance" tab
3. Click the "Load profile" button
4. Select your `.json` file
5. View the trace in Chrome's performance profiler

### Using nCompass VSCode Extension

1. Install the [nCompass VSCode Extension](https://docs.ncompass.tech/ncprof/quick-start)
2. Open the `.json` trace file in VSCode
3. The extension will automatically display the trace visualization

## Files

- `convert_nsys.py`: Main conversion script
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `vllm_trace_profile.nsys-rep`: Example nsys report file (already included)

## Troubleshooting

### nsys Command Not Found

If you see an error about `nsys` command not found:
- Ensure NVIDIA Nsight Systems is installed
- Verify `nsys` is in your PATH: `which nsys` (Linux/Mac) or `where nsys` (Windows)
- On Linux, you may need to add the installation directory to PATH:
  ```bash
  export PATH=$PATH:/usr/local/cuda/bin
  ```

### Import Errors

If you see import errors for `ncompass`:
- Ensure the virtual environment is activated
- Verify installation: `pip list | grep ncompass`
- Reinstall if needed: `pip install --upgrade ncompass>=0.1.7`

### File Not Found Errors

- Ensure the input file exists in the same directory as the script
- Use absolute paths if needed: `python convert_nsys.py --input /path/to/file.nsys-rep`

## References

- [nCompass SDK Documentation](https://docs.ncompass.tech)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [Perfetto Trace Viewer](https://ui.perfetto.dev)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)

