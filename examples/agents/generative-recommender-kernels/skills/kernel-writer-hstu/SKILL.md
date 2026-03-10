---
name: hstu-kernel-writer
description: >
   Write a correct and high performance CUDA HSTU attention kernel from scratch.
   Load when working on kernel writing, building, testing, profiling, or
   performance optimization in the hstu_fa_kernel/ environment.
---

# AI Kernel Writer — HSTU Attention from Scratch

You are an expert CUDA GPU kernel engineer. 
Your goal is to **write a correct and highly optimized HSTU attention kernel from scratch** in CUDA.
You may use CUTLASS libraries.

[CRITICAL] The output must be correct. Performant code is meaningless without correct code.
[CRITICAL] If you have access to the ncompass and knowledge_base MCP, use them extensively as they
are there to augment your reasoning. Ask questions, and iterate back and forth with those agents to
come to the best solution you can.
If you feel like you have exhausted all attempts and are going in circles, you can stop.

## Success Criteria

1. **Correctness**: `python hstu_fa_kernel/test_correctness.py` passes (atol=1e-1 default, atol=5e-2 with `--strict`)
2. **Performance**: Atleast 10x faster than the PyTorch reference latency on `python hstu_fa_kernel/bench.py`

## Environment Overview

The hstu_fa_kernel/ infrastructure sets up the problem of writing a CUDA HSTU attention kernel from scratch. The reference implementation is the PyTorch `pytorch_hstu_mha()` function.

- **Scratch kernel** (`hstu_fa_kernel/kernel/`): Your editable kernel code. Builds as `hstu_ai_optimized` package, registers `torch.ops.hstu_ai_optimized.*`.
[CRITICAL] These are the only files you can edit and any changes you make that are "hacks" and try
to game the way the kernel is being profiled for correctness of performance will be considered
invalid.
- **Reference**: `pytorch_hstu_mha()` from `hstu_fa_kernel/reference/pt_hstu_attention.py` — a pure PyTorch implementation.

### Directory Layout

```
hstu_fa_kernel/
    setup.py                # Builds kernel/ as hstu_ai_optimized package
    build.py                # Build the kernel
    bench.py                # Benchmark PyTorch ref vs scratch kernel
    test_correctness.py     # Compare outputs for correctness
    profile_ncu.py          # NCU profiling entry point
    profile_ncu_runner.py   # NCU wrapper script
    kernel/                 # Git-tracked editable kernel sources
        flash_fwd_launch_template.h   # *** YOUR MAIN FILE — replace the stub ***
        flash_api.cpp       # Op registration (hstu_ai_optimized namespace)
        flash_api_cpu.cpp   # Schema defs (hstu_ai_optimized namespace)
        flash_common.cpp    # hstu_mha_fwd host-side implementation
        flash_common_cpu.cpp
        flash.h             # Flash_fwd_params struct definition
        static_switch.h     # Template dispatch macros
        tile_size.h         # Tile size configuration
        instantiations/     # Forward-only sm90 .cu files
    reference/              # Local PyTorch reference (self-contained)
        __init__.py
        pt_hstu_attention.py  # pytorch_hstu_mha() — pure PyTorch reference
    baselines/              # Saved benchmark JSON
    ncu_reports/            # NCU report files
```

### Kernel Architecture

The call chain is:
1. Python calls `torch.ops.hstu_ai_optimized.hstu_mha_fwd(...)`
2. → `flash_api.cpp` dispatches to `hstu::hstu_mha_fwd()` in `flash_common.cpp`
3. → `flash_common.cpp` validates inputs, allocates output, calls `run_mha_fwd()`
4. → `run_mha_fwd()` dispatches by dtype/headdim to `hstu::run_mha_fwd_<Arch, T, HeadDim, Softmax>()`
5. → `flash_fwd_launch_template.h` contains `run_mha_fwd_()` — **this is where your kernel lives**
6. → Instantiated in `instantiations/flash_fwd_hdim128_bf16_softmax{true,false}_sm90.cu`

The stub in `flash_fwd_launch_template.h` currently just zeros the output. You replace it with the real attention computation.

### Files You Edit

- `kernel/flash_fwd_launch_template.h` — Your main kernel file
- Any new header files you create in `kernel/` (e.g., `mainloop_fwd.h`, `softmax.h`, `mask.h`, etc.)

### Files You Do NOT Edit

- `flash_api.cpp` / `flash_api_cpu.cpp` — Op registration boilerplate
- `flash_common.cpp` — Host-side setup, param validation, output allocation
- `flash.h` — `Flash_fwd_params` struct (read this to understand available params)
- `instantiations/*.cu` — Template instantiation files

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

At the very start of a session, before doing anything else, obtain a session ID. If a `.session_id` file exists in the working directory (created by the `setup-agent-run` script), read it. Otherwise, generate a new one:
```bash
if [ -f .session_id ]; then
    SESSION_ID=$(cat .session_id)
else
    SESSION_ID=$(openssl rand -hex 4)
    echo "$SESSION_ID" > .session_id
fi
echo "Session ID: $SESSION_ID"
```
Store this value and reuse it everywhere a session identifier is needed (docker tag, branch name, etc.).

### Start container and exec shell
[CRITICAL] You must tag the docker container you're creating with the session ID as we can have
multiple parallel sessions going on. You don't want to be working on another session's container.
```bash
python3 nc_pkg.py --run --tag "$SESSION_ID" --ncompass-dir ../../../ncompass
```

This starts the container, installs dependencies, installs ncompass, and drops you into a shell.

## Building the Kernel

[CRITICAL] Working directory in the docker container is same as the host.

### Build the scratch kernel

```bash
python hstu_fa_kernel/build.py
```
Verify: `python -c "import torch; import hstu_ai_optimized._C; print('OK')"`

### Build flags

The build uses these defaults (forward-only, BF16 hdim128 SM90):

| Flag | Value | Effect |
|------|-------|--------|
| `DISABLE_BACKWARD` | TRUE | No backward pass compilation |
| `DISABLE_FP16` | TRUE | BF16 only |
| `DISABLE_HDIM128` | FALSE | Enable hdim=128 |
| `DISABLE_HDIM{64,96,192,256}` | TRUE | Disable other head dims |
| `DISABLE_SM80` | TRUE | SM90 only |

### After editing kernel source

Just re-run `python hstu_fa_kernel/build.py`. Ninja will incrementally rebuild only changed files.

## Testing Correctness and Performance

### Correctness test

```bash
python hstu_fa_kernel/test_correctness.py
python hstu_fa_kernel/test_correctness.py --strict
python hstu_fa_kernel/test_correctness.py --batch-size 64 --max-seq-len 128
```

Tests three configurations: (causal, no-softmax), (causal, softmax), (non-causal, no-softmax).
Default tolerances: atol=1e-1, rtol=5e-2. With `--strict`: atol=5e-2.

### Benchmark

```bash
python hstu_fa_kernel/bench.py
python hstu_fa_kernel/bench.py --save-baseline hstu_fa_kernel/baselines/reference.json
python hstu_fa_kernel/bench.py --compare-baseline hstu_fa_kernel/baselines/reference.json
python hstu_fa_kernel/bench.py --batch-size 256 --max-seq-len 512 --iterations 200
```

Reports median/mean/min latency and estimated TFLOPS for both PyTorch reference and scratch kernel.

### Current target

- **Kernel**: HSTU attention forward (sm90, bf16, hdim128, causal, jagged)
- **Default benchmark config**: B=512, S=256, H=4, D=128, causal, no-softmax, jagged uniform
- **Metric**: Kernel latency (ms)

## Profiling with NCU

```bash
python hstu_fa_kernel/profile_ncu_runner.py                                 # default config
python hstu_fa_kernel/profile_ncu_runner.py -o my_report                    # custom output name
python hstu_fa_kernel/profile_ncu_runner.py -- --batch-size 256             # custom kernel args
```

Reports are saved in `hstu_fa_kernel/ncu_reports/`.

## Version Control

Track every kernel change as a separate commit so we can trace which edit produced which result.

1. **At the start of a session**, create a new branch from the current HEAD using the session ID generated earlier:
   ```bash
   git checkout -b kernel-write/$SESSION_ID
   ```

2. **After each kernel edit**, commit the changed files with a short summary of what was changed and why:
   ```bash
   git add hstu_fa_kernel/kernel/
   git commit -m "description of the change"
   ```
   The commit message should briefly describe what was modified (e.g., "implement naive global-memory attention", "add shared memory tiling for Q/K", "implement online softmax for softmax heads").

3. **Include test results** in the commit message body when available — whether the kernel was correct and the performance relative to the baseline.

## Key Constraints

- Always rebuild (`python hstu_fa_kernel/build.py`) after editing kernel sources before testing
- Always run `python hstu_fa_kernel/test_correctness.py` after edits to verify correctness
- Use `python hstu_fa_kernel/bench.py --compare-baseline hstu_fa_kernel/baselines/reference.json` to track progress
- The reference PyTorch code in `hstu_fa_kernel/reference/` must NEVER be modified
- Only forward pass is supported (backward is disabled)
- The CUTLASS/CuTe headers are already on the include path — just `#include` them when needed
