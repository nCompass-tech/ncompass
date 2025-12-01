# Nsight Systems Example: Profiling and Converting nsys Reports

This example demonstrates how to profile PyTorch neural network training using NVIDIA Nsight Systems (nsys) and convert the profiling reports to Chrome trace JSON format for visualization in the nCompass VSCode extension or other Chrome trace viewers.

## What This Example Does

This example shows how to:
- **Profile PyTorch training** using `nsys profile` to generate `.nsys-rep` files
- **Convert nsys reports** to SQLite format using the `nsys` CLI tool
- **Use the nCompass SDK** to convert SQLite databases to Chrome trace JSON format
- **Visualize GPU profiling data** in the nCompass VSCode extension
- **View traces** with GPU kernels, NVTX markers, CUDA API calls, and thread scheduling

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** (required)
- **NVIDIA Nsight Systems CLI** (`nsys` command) installed and available in your PATH
  - Download from: [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
  - Verify installation: `nsys --version`
- **VSCode** with the [nCompass extension](https://marketplace.visualstudio.com/items?itemName=nCompassTech.ncprof-vscode) installed (recommended for viewing traces)
- **CUDA-capable GPU** (for generating nsys reports, not required for conversion)

## Quick Start: Profile and Convert in One Command

The fastest way to get started is using `main.py`, which handles profiling and conversion in a single command:

```bash
# Profile the SimpleNet model and auto-convert to Chrome trace
python main.py --convert

# Profile with custom parameters
python main.py --epochs 20 --hidden-size 1024 --convert

# Just profile (no conversion)
python main.py --output my_profile
```

This will:
1. Run the SimpleNet training under `nsys profile`
2. Generate a `.nsys-rep` profiling report
3. Optionally convert to Chrome trace JSON (with `--convert`)

## Step-by-Step Guide

### Step 1: Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3.11 -m venv venv-nsys-example

# Activate the virtual environment
source venv-nsys-example/bin/activate  # On Windows: venv-nsys-example\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> ⚠️ **Troubleshooting**: If you encounter issues with `ncompasslib` or `pydantic`, ensure you're running Python 3.11 and have `Pydantic>=2.0` installed.

### Step 2: Verify nsys Installation

Ensure the `nsys` CLI tool is installed and accessible:

```bash
# Check if nsys is installed
nsys --version

# If not found, check common installation paths
which nsys  # Linux/Mac
where nsys  # Windows
```

If `nsys` is not found, you may need to:
- Install NVIDIA Nsight Systems from the [official website](https://developer.nvidia.com/nsight-systems)
- Add the installation directory to your PATH:
  ```bash
  export PATH=$PATH:/usr/local/cuda/bin
  # Or wherever nsys is installed
  ```

### Step 3: Profile with main.py (Recommended)

The easiest way to generate an nsys report is using `main.py`:

```bash
# Basic profiling with default parameters
python main.py

# Profile with custom training parameters
python main.py --epochs 20 --hidden-size 1024

# Profile and auto-convert to Chrome trace JSON
python main.py --convert

# Specify output name and trace types
python main.py --output my_profile --trace-types cuda,nvtx,osrt,cudnn
```

#### main.py CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 10 |
| `--hidden-size` | Hidden layer size for the network | 512 |
| `--output`, `-o` | Base name for output files | Auto-generated with timestamp |
| `--trace-types`, `-t` | Trace types: cuda,nvtx,osrt,cudnn,cublas | cuda,nvtx,osrt |
| `--convert`, `-c` | Auto-convert to Chrome trace JSON | False |
| `--no-force` | Don't overwrite existing output files | False |

### Step 3b: Alternative - Use Existing nsys Report

If you already have an `.nsys-rep` file, you can skip profiling:

**Option A: Use the example file**
- The example includes a test file at `test_files/test_trace.nsys-rep`
- Or use any existing `.nsys-rep` file you have

**Option B: Generate manually with nsys**
If you want to profile your own application directly:

```bash
# Profile a Python script
nsys profile --trace=cuda,nvtx,osrt python your_script.py

# Profile with specific options
nsys profile --trace=cuda,nvtx,osrt --output=my_profile python your_script.py
```

### Step 4: Convert nsys Report to Chrome Trace

Run the conversion script with your `.nsys-rep` file:

```bash
# Convert the default example file (if it exists)
python convert_nsys.py

# Or convert a specific file
python convert_nsys.py --input test_files/test_trace.nsys-rep

# Specify custom output name
python convert_nsys.py --input my_profile.nsys-rep --output my_trace
```

**What happens:**
1. **Step 1**: Converts `.nsys-rep` → `.sqlite` using the `nsys export` command
2. **Step 2**: Converts `.sqlite` → `.json` using the nCompass SDK
3. Generates both intermediate SQLite file and final Chrome trace JSON file

**Expected output:**
```
================================================================================
Running all steps: sqlite -> chrome
================================================================================
--------------------------------------------------------------------------------
Converting nsys report to SQLite...
Input: test_files/test_trace.nsys-rep
Output: test_trace.sqlite
SQLite conversion completed successfully!
SQLite file saved as: test_trace.sqlite
--------------------------------------------------------------------------------
Converting SQLite to Chrome trace format...
Input: test_trace.sqlite
Output: test_trace.json
Chrome trace conversion completed successfully!
Chrome trace file saved as: test_trace.json
--------------------------------------------------------------------------------
All conversions completed!
SQLite file: test_trace.sqlite
Chrome trace file: test_trace.json
```

### Step 5: View Traces in VSCode

1. **Open the trace file**: In VSCode, navigate to the generated `.json` file
2. **Open with trace viewer**: Right-click on the `.json` file → **Open With...** → **GPU Trace Viewer**
3. **Explore the trace**:
   - See GPU kernel executions
   - View NVTX markers and user annotations
   - Analyze CUDA API calls
   - Examine thread scheduling and OS runtime events

## Command-Line Options

### Basic Options

- `--input`, `-i`: Input nsys report file (`.nsys-rep`) or SQLite file (`.sqlite`) if using `--step chrome`. Default: `vllm_trace_profile.nsys-rep`
- `--output`, `-o`: Base name for output files (without extensions). If not specified, uses input filename without extension.
- `--step`, `-s`: Which step to run: `sqlite` (nsys-rep → sqlite), `chrome` (sqlite → json), or `all` (run all steps). Default: `all`

### Examples

```bash
# This runs the program on the default file (test_files/test_trace.nsys-rep)
python convert_nsys.py

# Convert specific file
python convert_nsys.py --input my_profile.nsys-rep

# Custom output name
python convert_nsys.py --input my_profile.nsys-rep --output my_trace

# Run only SQLite conversion step
python convert_nsys.py --step sqlite --input my_profile.nsys-rep

# Run only Chrome trace conversion (requires existing SQLite file)
python convert_nsys.py --step chrome --input my_profile.sqlite
```

## Understanding the Conversion Process

### Step 1: nsys-rep → SQLite

The script uses the `nsys export` command to convert the binary nsys report to SQLite format:

```bash
nsys export --type sqlite --include-json true --force-overwrite true -o output.sqlite input.nsys-rep
```

The SQLite database contains structured profiling data that can be programmatically queried and analyzed.

### Step 2: SQLite → Chrome Trace JSON

The nCompass SDK reads the SQLite database and converts it to Chrome trace format:

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

### Trace Contents

The generated Chrome trace JSON includes:
- **CUDA kernels**: GPU kernel execution events with timing information
- **NVTX markers**: User-defined annotations and ranges
- **CUDA API calls**: Runtime API events (memory allocation, kernel launches, etc.)
- **OS runtime events**: System-level events
- **Thread scheduling**: CPU thread scheduling information

## Output Files

### SQLite File (`.sqlite`)

The intermediate SQLite database contains structured profiling data exported from the nsys report. This format allows for:
- Programmatic querying and analysis
- Custom processing and filtering
- Integration with other tools

### Chrome Trace JSON (`.json`)

The final Chrome trace JSON file is compatible with:
- **nCompass VSCode Extension**: View traces directly in VSCode (recommended)
- **Perfetto UI**: [ui.perfetto.dev](https://ui.perfetto.dev) - Upload and visualize online
- **Chrome DevTools**: Open Chrome DevTools → Performance tab → Load profile

## Use Cases

### Use Case 1: End-to-End Profiling Workflow

Profile a model and view the trace in one workflow:

```bash
# Step 1: Profile and convert
python main.py --epochs 20 --hidden-size 1024 --convert --output my_training

# Step 2: Open my_training.json in VSCode with nCompass extension
# Right-click → Open With... → GPU Trace Viewer
```

### Use Case 2: Convert Existing nsys Reports

If you have existing `.nsys-rep` files from previous profiling sessions:

```bash
python convert_nsys.py --input /path/to/existing_profile.nsys-rep --output converted_trace
```

### Use Case 2: Batch Conversion

Convert multiple nsys reports:

```bash
for file in *.nsys-rep; do
    python convert_nsys.py --input "$file" --output "${file%.nsys-rep}"
done
```

### Use Case 3: Two-Step Conversion

If you want to inspect the SQLite database before converting:

```bash
# Step 1: Convert to SQLite
python convert_nsys.py --step sqlite --input my_profile.nsys-rep

# Inspect SQLite file (optional)
sqlite3 my_profile.sqlite "SELECT name FROM sqlite_master WHERE type='table';"

# Step 2: Convert to Chrome trace
python convert_nsys.py --step chrome --input my_profile.sqlite
```

## Docker Setup (Optional)

If you prefer to use Docker or don't have nsys installed locally, you can run the example in a containerized environment:

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (for GPU support, if needed)

### Quick Start

```bash
# Build and run the container
python nc_pkg.py --build --run
```

This will:
- Build a Docker image with nsys CLI and all dependencies pre-installed
- Start a container with GPU access
- Drop you into an interactive shell inside the container

### Docker Commands

```bash
# Build the image
python nc_pkg.py --build

# Run the container (interactive shell)
python nc_pkg.py --run

# Run a command in the container
python nc_pkg.py --exec "python convert_nsys.py --input test_files/test_trace.nsys-rep"

# Stop and remove the container
python nc_pkg.py --down
```

### Docker Notes

- The container mounts the current directory to `/workspace`
- Output files are saved in the mounted directory and accessible from the host
- GPU access is enabled via NVIDIA Docker runtime
- The container includes nsys CLI pre-installed, so no manual installation needed

## File Structure

- `main.py`: Nsys profiling script - runs training under `nsys profile` and optionally converts traces
- `simplenet.py`: Simple neural network model and training function
- `convert_nsys.py`: Conversion script for existing nsys reports (nsys-rep → SQLite → JSON)
- `requirements.txt`: Python dependencies (ncompass SDK)
- `nc_pkg.py`: Docker helper script
- `test_files/test_trace.nsys-rep`: Example nsys report file
- `README.md`: This file

## Additional Resources

- **[nCompass VSCode Extension Documentation](https://docs.ncompass.tech/ncprof/quick-start)** - Complete guide to using the extension
- **[NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)** - Official nsys documentation

## Support

For questions or issues:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- Open an issue on [GitHub](https://github.com/ncompass-tech/ncompass/issues)
