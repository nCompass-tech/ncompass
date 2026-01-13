# Nsight Compute Example: Profiling PyTorch Inference

This example demonstrates how to profile PyTorch neural network inference using NVIDIA Nsight Compute (ncu) and generate detailed CSV profiling reports with kernel-level metrics.

## What This Example Does

This example shows how to:
- **Profile PyTorch inference** using `ncu` to generate CSV reports
- **Collect detailed GPU metrics** including timing, memory bandwidth, and floating-point operations
- **Filter profiling** by specific kernel names or NVTX ranges
- **Analyze kernel performance** with detailed, per-kernel metric breakdowns

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (required)
- **NVIDIA Nsight Compute CLI** (`ncu` command) installed and available in your PATH
  - Download from: [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)
  - Verify installation: `ncu --version`
- **CUDA-capable GPU** (required for generating ncu reports)

## Quick Start: Profile in One Command

The fastest way to get started is using `main.py`, which handles the profiling setup and execution:

```bash
# Profile the SimpleNet model (generates a CSV report)
python main.py

# Profile with custom parameters
python main.py --iters 30 --hidden-dim 4096 --output my_profile

# Profile with fp16 precision
python main.py --precision fp16
```

This will:
1. Run SimpleNet inference under `ncu`
2. Collect default metrics (or custom metrics if specified)
3. Generate a `.csv` profiling report in the `.ncu_traces/` directory

## Step-by-Step Guide

### Step 1: Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create a virtual environment
python3 -m venv venv-ncu-example

# Activate the virtual environment
source venv-ncu-example/bin/activate  # On Windows: venv-ncu-example\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify ncu Installation

Ensure the `ncu` CLI tool is installed and accessible:

```bash
# Check if ncu is installed
ncu --version

# If not found, check common installation paths
which ncu  # Linux/Mac
where ncu  # Windows
```

If `ncu` is not found, you may need to:
- Install NVIDIA Nsight Compute from the [official website](https://developer.nvidia.com/nsight-compute)
- Add the installation directory to your PATH:
  ```bash
  export PATH=$PATH:/usr/local/cuda/bin
  # Or wherever ncu is installed
  ```

### Step 3: Profile with main.py

The `main.py` script provides a convenient interface for profiling the SimpleNet model:

```bash
# Basic profiling with default parameters
python main.py

# Profile with custom inference parameters
python main.py --batch 256 --iters 50 --hidden-dim 1024

# Specify output name
python main.py --output my_ncu_report

# Filter by kernel name
python main.py --kernel-name "regex:.*gemm.*"
```

#### main.py CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch` | Batch size for inference | 512 |
| `--iters` | Number of profiling iterations | 20 |
| `--warmup` | Number of warmup iterations | 5 |
| `--input-dim` | Input dimension for the model | 1024 |
| `--hidden-dim` | Hidden layer dimension | 2048 |
| `--output-dim` | Output dimension | 512 |
| `--precision` | Precision: fp32 or fp16 | fp32 |
| `--output`, `-o` | Base name for output files | Auto-generated with timestamp |
| `--kernel-name`, `-k` | Regex pattern to filter kernels | (all kernels) |

**Note:** Metrics are sourced from the ncompass library's default configuration (`ncompass.profile.config`), which includes a comprehensive set of timing, throughput, and floating-point operation metrics.

## Understanding the Profiling Process

### Metric Collection

The script collects a comprehensive set of metrics by default, including:
- **GPU Timing**: `gpu__time_duration.sum`
- **Throughput**: SM, DRAM, L2, and L1 cache utilization percentages
- **Compute Intensity**: Floating-point operations (FP32, FP16, FP64)
- **Memory Traffic**: DRAM bytes read/written

### NVTX Integration

The `main.py` script runs `ncu` with the `--nvtx` flag enabled, allowing it to capture NVTX markers defined in the application. This helps in correlating high-level application logic with specific GPU kernels.

### CSV Output

Unlike Nsight Systems, which generates binary reports, this example configures `ncu` to output results directly in CSV format. The CSV files are saved in a timestamped subdirectory under `.ncu_traces/`.

Each row in the CSV represents a kernel execution, and columns provide the requested metrics for that specific kernel.

## Docker Setup (Optional)

If you prefer to use Docker or want a pre-configured environment, use the provided `nc_pkg.py` helper script:

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Docker runtime (required for GPU profiling)

### Commands

```bash
# Build the image
python nc_pkg.py --build

# Run the container (interactive shell)
python nc_pkg.py --run

# Run a profiling command in the container
python nc_pkg.py --exec "python main.py --iters 10"

# Stop and remove the container
python nc_pkg.py --down
```

The container mounts the current directory to `/workspace`, making your results available on the host machine.

## File Structure

- `main.py`: Profiling script - handles `ncu` execution and CSV report generation
- `simplenet.py`: Simple neural network model and inference function
- `nc_pkg.py`: Docker helper script for building and running containers
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container definition with `ncu` pre-installed
- `README.md`: This file

## Additional Resources

- **[NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)** - Official ncu documentation
- **[nCompass Documentation](https://docs.ncompass.tech)** - Learn more about nCompass tools

## Support

For questions or issues:
- Check the [Documentation](https://docs.ncompass.tech)
- Visit the [Community Forum](https://community.ncompass.tech)
- Open an issue on [GitHub](https://github.com/ncompass-tech/ncompass/issues)
