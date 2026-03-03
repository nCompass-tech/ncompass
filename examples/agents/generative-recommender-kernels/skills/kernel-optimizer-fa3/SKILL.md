---
name: fa3-optimizer
description: >
   Iteratively optimize CUDA kernel source code for performance while
   maintaining correctness. Load when working on kernel editing, building,
   profiling, or performance optimization in the HSTU FA3 runner environment.
---

# AI Kernel Optimizer

You are required to keep editing the original source code of the kernel until you have a kernel that is as fast as you deem possible and correct in the output. 
If you feel like you have exhausted all attempts to make it faster and are going in circles, you can stop.

## Environment Overview

The runner infrastructure supports iterative optimization of the HSTU Flash Attention 3 forward kernel. Two copies of the kernel exist:
- **Optimized kernel** (`fa3/kernel/`): Editable copy. Builds as `hstu_runner` package, registers `torch.ops.hstu_opt.*`.

Both can be loaded simultaneously in the same Python process for A/B comparison.

### Directory Layout

```
fa3/
    setup.py                # Builds fa3/kernel/ as hstu_runner package
    build.sh                # Build optimized kernel
    build_reference.sh      # Build reference kernel
    bench.py                # Benchmark both kernels
    test_correctness.py     # Compare outputs for correctness
    profile_ncu.py          # NCU profiling entry point
    profile_ncu.sh          # NCU wrapper script
    kernel/                 # Git-tracked editable kernel sources
        *.h                 # 23 header files
        flash_api.cpp       # Op registration (hstu_opt namespace)
        flash_api_cpu.cpp   # Schema defs (hstu_opt namespace)
        flash_common.cpp    # hstu_mha_fwd implementation
        flash_common_cpu.cpp
        instantiations/     # Forward-only sm90 .cu files
    reference/
        setup.py            # Builds reference kernel as hstu package
        cpp_compat.patch    # C++ compat patches for generative-recommenders
    baselines/              # Saved benchmark JSON
    ncu_reports/            # NCU report files
```

### Kernel Source Management

The optimized kernel source in `fa3/kernel/` is **git-tracked directly**. It was originally derived from the reference source in `generative-recommenders/` with namespace changes (`hstu` -> `hstu_opt`) and C++ compatibility fixes already applied. Edit the files in `fa3/kernel/` directly.

## Kernel Source Code

### Files you typically do NOT edit

- `flash_api.cpp` / `flash_api_cpu.cpp` - Op registration boilerplate
- `flash_common.cpp` - Host-side setup, param validation
- `instantiations/*.cu` - Template instantiation (auto-generated pattern)

## Docker Environment

Set up the Docker dev environment before building kernels. Inside Docker, `python` and `pip` are on PATH — no virtualenv prefix needed.

### One-time setup

```bash
python3 nc_pkg.py --setup --docker-dir ../../docker
```

This creates a `docker/` symlink to the shared Docker infrastructure.

### Build the image
```bash
python3 nc_pkg.py --build
```

### Session ID

At the very start of a session, before doing anything else, generate a short unique session ID:
```bash
SESSION_ID=$(openssl rand -hex 4)
echo "Session ID: $SESSION_ID"
```
Store this value and reuse it everywhere a session identifier is needed (docker tag, branch name, etc.).

### Start container and exec shell
[CRITICAL] You must tag the docker container you're creating with the session ID as we can have
multiple parallel sessions going on. You don't want to be working on another optimizer's session.
```bash
python3 nc_pkg.py --run --tag "$SESSION_ID" --ncompass-dir ../../../ncompass
```

This starts the container, installs generative-recommenders dependencies, installs ncompass, and drops you into a shell.

## Building the Kernel
[CRITICAL] Working directory in the docker container is same as the host.

### Build optimized kernel

```bash
fa3/build.sh
```
Verify: `python -c "import hstu_runner._C; print('OK')"`

### Build reference kernel

```bash
fa3/build_reference.sh
```
Verify: `python -c "import hstu._C; print('OK')"`

### Build flags

The build uses these defaults (forward-only, BF16 hdim128 SM90):

| Flag | Value | Effect |
|------|-------|--------|
| `DISABLE_BACKWARD` | TRUE | No backward pass compilation |
| `DISABLE_FP16` | TRUE | BF16 only |
| `DISABLE_HDIM128` | FALSE | Enable hdim=128 |
| `DISABLE_HDIM{64,96,192,256}` | TRUE | Disable other head dims |
| `DISABLE_SM80` | TRUE | SM90 only |
| No `-DOSS_ENV` | — | Direct include paths |

### After editing kernel source

Just re-run `bash fa3/build.sh`. Ninja will incrementally rebuild only changed files.

## Testing Correctness and Performance

### Correctness test

```bash
python fa3/test_correctness.py
python fa3/test_correctness.py --batch-size 64 --max-seq-len 128
```

Tests three configurations: (causal, no-softmax), (causal, softmax), (non-causal, no-softmax). Tolerances: atol=5e-2, rtol=5e-2 (bf16).

### Benchmark

```bash
python fa3/bench.py
python fa3/bench.py --save-baseline fa3/baselines/reference.json
python fa3/bench.py --compare-baseline fa3/baselines/reference.json
python fa3/bench.py --batch-size 256 --max-seq-len 512 --iterations 200
```

Reports median/mean/min latency and estimated TFLOPS for both kernels.

### Current optimization target

- **Kernel**: HSTU FA3 forward (sm90, bf16, hdim128, causal, jagged)
- **Default benchmark config**: B=512, S=256, H=4, D=128, causal, no-softmax, jagged uniform
- **Metric**: Kernel latency (ms)

## Profiling with NCU

```bash
bash fa3/profile_ncu.sh                                 # default config
bash fa3/profile_ncu.sh -o my_report                    # custom output name
bash fa3/profile_ncu.sh -- --batch-size 256             # custom kernel args
bash fa3/profile_ncu.sh -- --kernel ref                 # profile reference
```

Reports are saved in `fa3/ncu_reports/`.

## Version Control

Track every kernel change as a separate commit so we can trace which edit produced which result.

1. **At the start of a session**, create a new branch from the current HEAD using the session ID generated earlier:
   ```bash
   git checkout -b kernel-opt/$SESSION_ID
   ```

2. **After each kernel edit**, commit the changed files with a short summary of what was changed and why:
   ```bash
   git add fa3/kernel/
   git commit -m "description of the change"
   ```
   The commit message should briefly describe what was modified (e.g., "increase tile size from 64 to 128 for better occupancy", "unroll inner loop to reduce register pressure").

3. **Include test results** in the commit message body when available — whether the kernel was correct and the performance relative to the baseline.

## Key Constraints

- Always rebuild (`bash fa3/build.sh`) after editing kernel sources before testing
- Always run `python fa3/test_correctness.py` after edits to verify correctness
- Use `python fa3/bench.py --compare-baseline fa3/baselines/reference.json` to track progress
- The reference kernel in `generative-recommenders/generative_recommenders/ops/cpp/hstu_attention/` must NEVER be modified
- Only forward pass is supported (backward is disabled)
