# Nsight Compute Example: Profiling PyTorch Inference

This example demonstrates how to profile specific regions of 
AI inference using NVIDIA Nsight Compute (ncu) with NVTX markers added via the 
nCompass VSCode extension.

## What This Example Does

This example shows how to:
- **Add NVTX markers** using the nCompass VSCode extension (no code changes required)
- **Profile PyTorch inference** using `ncu` to generate CSV reports
- **Filter profiling** by NVTX ranges to focus on specific code regions
- **Analyze kernel performance** with detailed, per-kernel metric breakdowns

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (required)
- **NVIDIA Nsight Compute CLI** (`ncu` command) installed and available in your PATH
  - Download from: [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
  - Verify installation: `ncu --version`
- **CUDA-capable GPU** (required for generating ncu reports)
- **nCompass VSCode Extension** - [Quick Start](https://docs.ncompass.tech)

## Quick Start

### Step 1: Install Example Dependencies (Local)

```bash
# Create and activate a virtual environment
python3 -m venv venv-ncu-example
source venv-ncu-example/bin/activate

# Install ncompass from the parent directory
pip install ../../../ncompass

# Install other dependencies
pip install -r requirements.txt
```

### Step 1 (alternative): Install Example Dependencies (Docker)

```bash
python nc_pkg.py --build --run
```
This puts you in a docker container with all the dependencies installed

### Step 2: Check ncu priveleges are correct
On your **host** machine, run the following command: 
```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```
This should output `RmProfilingAdminOnly: 0`. If this value is 1, you need to run the following
steps to have sufficient priveleges.
```bash
sudo rmmod nvidia_uvm
sudo rmmod nvidia_drm
sudo rmmod nvidia_modeset
sudo rmmod nvidia
sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0
```

### Step 3: Add NVTX Markers via VSCode Extension

Use the nCompass VSCode extension to add NVTX markers to the code you want to profile:

1. Open `simplenet.py` in VSCode/Cursor with the nCompass extension installed.
2. Highlight the code region you want to profile (e.g., the `forward` method or specific layers).
3. Press `Ctrl+.` (or `Cmd+.` on Mac) or click the lightbulb icon.
4. Select **"Add Region to Profile"** from the menu.
5. Choose **"NVTX"** as the marker type.

This creates a `config.json` file in the `.cache` directory at the top level of your VSCode
workspace.

For a video walkthrough, see: [Runtime Injection Tutorial](https://www.loom.com/share/1e526ff43fa44bfcaebe18a6f56052ff)

### Step 4: Run the Profiler

Set the required environment variables and run the profiler:

```bash
# Set environment variables
export NCOMPASS_CACHE_DIR=<path to the directory where the .cache/ dir was created in Step 3>
export NCOMPASS_PROFILER_TYPE=NVTX

# Run profiling
python main.py --ncompass-dir ../../../ncompass
```

This will:
1. Load your NVTX marker configuration from `.cache/.../config.json`
2. Run SimpleNet inference under `ncu` with NVTX filtering
3. Generate a `.csv` profiling report in the `.ncu_traces/` directory

### Step 4: View Results

The generated CSV file contains per-kernel metrics for the regions you marked. You can:
- Open the CSV directly to analyze kernel performance
- Load the trace in the nCompass VSCode extension for visualization [COMING SOON]

## Understanding the Workflow

### Runtime Injection (No Code Changes)

The nCompass SDK uses AST-level code injection to add profiling markers at runtime. This means:
- Your source code remains unchanged
- Markers are injected when the module is imported
- Configuration is stored in `.cache/.../config.json`

### NVTX Integration

The profiler runs `ncu` with `--nvtx` and `--nvtx-include` flags to:
- Capture only the regions you marked in the extension
- Filter out noise from unmarked code
- Correlate high-level code regions with GPU kernels

### CSV Output

Results are saved as CSV files in `.ncu_traces/`. Each row represents a kernel execution with columns for timing, memory bandwidth, and compute metrics.

## File Structure

- `main.py` - Profiling script that invokes `ncu` and generates reports
- `simplenet.py` - Neural network model (target for NVTX markers)
- `runners/run_simplenet.py` - Runner that loads nCompass rewrites before execution
- `nc_pkg.py` - Docker helper script
- `requirements.txt` - Python dependencies

## Additional Resources

- [Runtime Injection Video Tutorial](https://www.loom.com/share/1e526ff43fa44bfcaebe18a6f56052ff)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [nCompass Documentation](https://docs.ncompass.tech)
