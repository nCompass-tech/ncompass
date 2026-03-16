---
name: hstu-kernel-writer
description: >
   Write a correct and high performance HSTU attention kernel from scratch.
   Load when working on kernel writing, building, testing, profiling, or
   performance optimization in the hstu_fa_kernel/ environment.
---

# AI Kernel Writer — HSTU Attention from Scratch

You are an expert GPU kernel engineer. 
Your goal is to **write a correct and highly optimized HSTU attention kernel from scratch**.
You may use CUTLASS libraries.

[CRITICAL] The output must be correct. Performant code is meaningless without correct code.
[CRITICAL] If you have access to the ncompass and knowledge_base MCP, use them extensively as they
are there to augment your reasoning. Ask questions, and iterate back and forth with those agents to
come to the best solution you can.
[CRITICAL] Use the plan that the knowledge_base planner provides and reason based on that. DO NOT
TRY TO COME UP WITH YOUR OWN STRATEGY.
[CRITICAL] You are not allowed to run any of the fill_in_todo tool calls in parallel. You have to
get context for a TODO - implement that and then move on to the next.
[CRITICAL] When the KB plan provides WGMMA/TMA API calls, COPY THEM VERBATIM into your kernel.
Do not simplify, do not "start simple and optimize later." The KB code IS the simple version —
it is the minimum viable WGMMA kernel. Replacing WGMMA with naive global memory loops or
replacing TMA with simple loads is NOT a valid simplification — it produces a kernel that is
100x slower and defeats the entire purpose.
[CRITICAL] Do NOT reason about whether WGMMA is "too complex" or whether TMA "adds significant
complexity." The KB has already validated this approach against real reference implementations.
Your job is to ADAPT the reference code to the specific task, not to evaluate whether to use it.
If you find yourself thinking "let me start with a simpler approach first" — STOP. That is the
failure mode. The KB skeleton IS the starting point.
[CRITICAL] When fill_in_todo returns reference source code, copy the API call patterns directly.
The reference code uses those specific APIs for a reason — they are the correct way to achieve
performance on this GPU architecture.
If you feel like you have exhausted all attempts and are going in circles, you can stop.

## Success Criteria

1. **Correctness**: `python hstu_fa_kernel/test_correctness.py` passes (atol=1e-1 default, atol=5e-2 with `--strict`)
2. **Performance**: Atleast 30x faster than the PyTorch reference latency on `python hstu_fa_kernel/bench.py`

## Environment Overview

The hstu_fa_kernel/ infrastructure sets up the problem of writing a HSTU attention kernel from scratch. The reference implementation is the PyTorch `pytorch_hstu_mha()` function.

- **Scratch kernel** (`hstu_fa_kernel/kernel/`): Your editable kernel code. Builds as `hstu_ai_optimized` package, registers `torch.ops.hstu_ai_optimized.*`.
[CRITICAL] Only files listed in "Files You Edit" below may be modified. Any changes that are
"hacks" to game profiling, correctness, or performance measurement will be considered invalid.
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

**Device-side (kernel code):**
- `kernel/flash_fwd_launch_template.h` — Your main kernel file (TMA descriptor creation, kernel launch)
- Any new header files you create in `kernel/` (e.g., `mainloop_fwd.h`, `softmax.h`, `mask.h`, etc.)

**Host-side (required for TMA and advanced optimizations):**
- `kernel/flash.h` — Parameter structs (`Flash_fwd_params`, `Qkv_params`) — add TMA descriptor fields here to pass descriptors from host to kernel
- `kernel/flash_common.cpp` — Host-side parameter setup — memory alignment for TMA (16-byte aligned), empty-sequence guards, `cudaFuncSetAttribute` for dynamic shared memory
- `kernel/flash_api.cpp` — Python-to-C++ interface — update signatures if new parameters are added

**Configuration:**
- `kernel/tile_size.h` — Tile/block size config that determines shared memory layouts (TMA tile dimensions must match)

### Files You Do NOT Edit

- `flash_api_cpu.cpp` / `flash_common_cpu.cpp` — CPU schema defs (no kernel logic)
- `instantiations/*.cu` — Template instantiation files
- `setup.py` / `build.py` — Build infrastructure
- `bench.py` / `test_correctness.py` / `profile_ncu*.py` — Test and profiling harnesses
- `reference/` — PyTorch reference implementation

## Session ID

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
Store this value and reuse it everywhere a session identifier is needed (branch name, etc.).

## Building the Kernel

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
   git init
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
