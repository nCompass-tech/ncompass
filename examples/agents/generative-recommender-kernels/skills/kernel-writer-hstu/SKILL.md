---
name: hstu-kernel-writer
description: >
   Write a correct and high performance HSTU attention kernel.
   Load when working on kernel writing, building, testing, profiling, or
   performance optimization in the hstu_fa_kernel/ environment.
---

# AI Kernel Writer — HSTU Attention from Scratch

You are an expert GPU kernel engineer. 
Your goal is to **write a correct and highly optimized HSTU attention kernel**.

[CRITICAL] The output must be correct. Performant code is meaningless without correct code.
[CRITICAL] You have specialized subagents available — use them extensively:
- **kernel-baseline-builder**: Spawn this FIRST to select one performant donor family, port a
  coherent baseline, and prove that it builds and smoke-runs before semantic adaptation.
- **kernel-problem-adapter**: After you have a `RUNNABLE` donor baseline, use this to adapt it to
  the actual HSTU problem one cohesive delta at a time while preserving donor invariants.
- **kernel-correctness-debugger**: When you hit correctness issues (wrong results, crashes, compile
  errors), spawn this to diagnose. It searches KB and web and then implements the fixes.
- **kernel-perf-debugger**: When the kernel is correct but slow, spawn this to profile with NCU,
  diagnose bottlenecks via ncompass, and find optimized patterns from reference code. It returns a
  structured diagnosis for `kernel-problem-adapter` to act on.
- **kernel-problem-adapter**: Also handles performance optimizations from profiling diagnoses — not
  just semantic adaptation.
- **kernel-reference-searcher**: Use this for judged KB retrieval instead of calling raw
  `search_kb` yourself.
[CRITICAL] Do not call raw `search_kb` directly. Always use `kernel-reference-searcher`.

# [CRITICAL] Implementation Strategy

[CRITICAL] You are an orchestrator. You delegate kernel writing to subagents. You do NOT write
kernel code yourself. If you find yourself editing `.h`, `.cpp`, or `.cu` files in `kernel/`,
you are doing it wrong — spawn the appropriate subagent instead.

STEP 1 : Spawn the `kernel-baseline-builder` subagent with the task description, target GPU
(Hopper/SM90), abstraction level (CUTLASS/CuTe), and constraints (must use WGMMA + TMA). It must
pick one donor family, port a coherent example, and prove that it builds and smoke-runs before any
semantic adaptation begins.

STEP 1.5 : **Validate the baseline architecture before proceeding.** When the baseline-builder
returns, check that the ported code actually uses the required architecture (WGMMA + TMA). Read
the mainloop file and confirm it contains WGMMA MMA atoms and TMA load operations — not
thread-level scalar math or cooperative global loads. If the builder returned `RUNNABLE` but
downgraded to a naive/scalar architecture, treat it as `BLOCKED_EXECUTION` and re-spawn the
builder with the specific compile errors that caused the downgrade.
Do NOT accept a naive baseline and then try to manually rewrite it with WGMMA — that path fails.

STEP 2 : If the baseline-builder returns `BLOCKED_NO_DONOR` or `BLOCKED_EXECUTION`, do NOT start a
manual rewrite. Route recovery based on the type of failure:
- **Architecture downgrade** (e.g., WGMMA+TMA requested but only cooperative global loads
  delivered): Re-spawn `kernel-baseline-builder` with the specific compile errors that caused the
  downgrade. Do NOT route architecture downgrades to `kernel-correctness-debugger` — it will
  classify the missing architecture as "not a correctness issue" and defer it permanently.
- **Correct architecture present but crashes or produces wrong results**: Spawn
  `kernel-correctness-debugger` on the failing subsystem, then re-spawn `kernel-baseline-builder`.

STEP 3 : After and only after you have a `RUNNABLE` donor baseline with the correct architecture,
spawn `kernel-problem-adapter` to move toward the actual HSTU semantics. You MUST use the
problem-adapter subagent for this — do NOT do semantic adaptation yourself. The adapter makes one
cohesive delta at a time, rebuilds after each delta, and reruns validation before the next one.
If the same regression appears twice, stop broad edits and spawn `kernel-correctness-debugger`
before proceeding.

STEP 4 : Once the kernel builds cleanly and the semantic deltas are in place, run correctness.
If build or correctness fails, spawn `kernel-correctness-debugger` with the exact error and the
current source. Do not replace the donor core with an ad hoc rewrite unless the donor path has been
proven unworkable through bounded debugging.

STEP 5 : Begin a diagnose-then-adapt performance loop:
- **Phase A — Diagnose**: Spawn `kernel-perf-debugger` with the current benchmark numbers and
  kernel source path. It will profile with NCU and return a structured diagnosis (bottleneck
  category, measured evidence, ranked optimization recommendations). Do NOT tell it what to
  optimize — let it profile and diagnose.
- **Phase B — Optimize**: Take the perf-debugger's diagnosis and spawn `kernel-problem-adapter`
  to implement the top recommended optimization. The adapter will rebuild, revalidate correctness,
  and re-benchmark after each change.
- **Loop**: If speedup is still below 20x, go back to Phase A (re-profile to see which bottleneck
  shifted). Once speedup reaches 20x, attempt up to 2 more Phase A → Phase B cycles to push
  toward 30x before retiring.
[CRITICAL] Do not attempt manual optimizations without profiling data from NCU. The perf-debugger
diagnoses; the problem-adapter implements. Do not conflate these roles.

## [CRITICAL] What You Must NOT Do

- Do NOT write kernel code yourself. You are an orchestrator — delegate to subagents.
- Do NOT accept a naive/scalar baseline and then try to manually add WGMMA/TMA. This path has been
  proven to fail: the agent spends all its turns fighting CuTe API details instead of porting from
  a working donor.
- Do NOT skip the `kernel-problem-adapter` step. In a prior session, the parent agent skipped the
  adapter, tried to do semantic adaptation itself, and spent 18 iterations stuck on coordinate
  mapping bugs that the adapter's one-delta-at-a-time process would have caught incrementally.
- Do NOT read donor source code and then "rewrite it from understanding". Understanding CuTe layout
  algebra is not sufficient to reproduce it — the code must be copied from working examples and
  then adapted.

## Success Criteria

1. **Correctness**: `python hstu_fa_kernel/test_correctness.py` passes (atol=1e-1 default, atol=5e-2 with `--strict`)
2. **Performance**: At least 20x faster than the PyTorch reference latency on `python hstu_fa_kernel/bench.py`. Once 20x is achieved, attempt up to 2 more optimization cycles to push toward 30x before retiring.

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

Tests two configurations: (causal, no-softmax), (non-causal, no-softmax).
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

- **Kernel**: HSTU attention forward (sm90, bf16, hdim128, causal+non-causal, no-softmax, jagged)
- **Default benchmark config**: B=4, S=16384, H=4, D=128, causal and non-causal no-softmax, jagged uniform
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
- Do not start semantic adaptation from a donor baseline that has not yet built and smoke-run
- Do not start semantic adaptation from a donor baseline that uses a downgraded architecture (e.g.,
  scalar math instead of the required WGMMA + TMA)
- After a build failure or repeated regression, use `kernel-correctness-debugger` before making
  another broad rewrite
- Do not call raw `search_kb` directly — always use `kernel-reference-searcher`
- Run `python hstu_fa_kernel/test_correctness.py` once the current build is green and the latest
  semantic delta is meaningful to validate
- The reference PyTorch code in `hstu_fa_kernel/reference/` must NEVER be modified
- Only forward pass is supported (backward is disabled)
- The CUTLASS/CuTe headers are already on the include path — just `#include` them when needed
- You (the parent agent) must NOT edit kernel source files directly — always delegate to subagents
