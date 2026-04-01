---
name: system-optimizer-setup
description: >
  One-time setup for system-level optimization sessions. Environment setup,
  baseline capture, MCP preflight, and structured codebase analysis. Run this
  before starting the optimization loop.
---

# System Optimizer — Session Setup

Run this skill once at the start of a system-level optimization session.
It handles environment setup, baseline measurement, and — critically —
a structured codebase analysis that informs your optimization hypotheses.

## Step 1: Environment Setup

Invoke the `/baseline-setup` skill and follow its instructions exactly. This
patches the source tree, builds the HSTU kernel, and installs dependencies.

## Step 2: Session ID and Branch

```bash
if [ -f .session_id ]; then
    SESSION_ID=$(cat .session_id)
else
    SESSION_ID=$(openssl rand -hex 4)
    echo "$SESSION_ID" > .session_id
fi
git checkout -b system-opt/$SESSION_ID
```

## Step 3: Create Notes Directory

```bash
mkdir -p .agent/notes model_runner/baselines model_runner/nsys_traces
```

## Step 4: MCP Preflight

Check which MCPs are available by calling each tool directly:

1. `mcp__ncompass__check_auth` (no arguments) — if it returns a result,
   ncompass is available
2. `mcp__knowledge_bank__search_kb` with query `"cuda graphs"` — if it
   returns results, KB is available

Record results in `state.json` under `mcps_available`.

## Step 5: Save Baseline

```bash
python model_runner/bench.py --max-seq-len 256 \
  --save-baseline model_runner/baselines/ref.json \
  --warmup-iters 10 --bench-iters 20
```

Read `model_runner/baselines/ref.json` for the baseline median.

## Step 6: Codebase Analysis

[CRITICAL] Before formulating any optimization hypotheses, read the model
source code thoroughly and document what you find. This step is what
separates effective optimization from trial-and-error.

Read these files in order:

1. `model_runner/run_model.py` — baseline forward pass, model setup,
   batch generation, warmup and profiling utilities
2. The model's `forward()`, `main_forward()`, `_user_forward()`,
   `_item_forward()` methods (in `generative-recommenders/`)
3. The HSTU transducer forward path
4. The preprocessing and merge logic
5. `model_runner/optimizations/__init__.py` — how optimization modes are
   discovered and applied

Write `.agent/notes/codebase_analysis.md` documenting:

```markdown
# Codebase Analysis

## Model architecture
- How does the forward pass decompose (preprocess → main_forward → postprocess)?
- How does the optimization mode system work?

## Synchronization points
- All .item() call sites and their purpose
- All .cpu(), .numpy(), or other GPU→CPU syncs

## Dtype conversions
- Where does fp32 ↔ bf16 conversion happen?
- Which weights are fp32 vs bf16?
- Are there redundant dtype casts in the forward path?

## Unused computation in eval/inference mode
- Training-only branches that still execute
- KV cache updates (are they used in this workload?)
- Gradient-related overhead

## Compile-friendly vs compile-hostile modules
- Which nn.Module children have clean tensor-in/tensor-out signatures?
- Which use .item(), dynamic shapes, custom Triton ops, or KJT operations?

## Kernel launch inventory
- Rough count of kernels per forward pass (from nsys or code reading)
- Which modules contribute the most launches?
```

## Step 7: Initial Profile

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=true \
  --cuda-graph-trace=node \
  -tcuda,nvtx \
  -o model_runner/nsys_traces/baseline \
  --force-overwrite=true \
  python -u model_runner/bench.py --max-seq-len 256 --mode baseline \
    --profile --bench-iters 3
```

If ncompass is available, analyze the trace:
```
mcp__ncompass__analyze_nsys_sqlite with the trace file
```

Write `.agent/notes/bottleneck.md` with the profiling analysis.

## Step 8: Initial Hypotheses

Based on the codebase analysis AND the profile, write
`.agent/notes/hypotheses.md` with a ranked list of optimization ideas.
Each hypothesis should reference specific findings from step 6.

## Step 9: Initialize State

Write `.agent/notes/state.json`:
```json
{
  "session_id": "<from .session_id>",
  "iteration": 0,
  "baseline_median_ms": <from ref.json>,
  "best_median_ms": <same>,
  "best_mode": "baseline",
  "best_commit": null,
  "current_bottleneck": "<from profiling>",
  "mcps_available": ["ncompass", "knowledge_bank"],
  "working_optimizations": [],
  "start_time": "<ISO timestamp>"
}
```

Write `.agent/notes/methodology.md` with benchmark methodology decisions.

## Step 10: Commit Setup

```bash
git add .agent/notes/ model_runner/baselines/
git commit -m "Session setup: baseline + codebase analysis"
```

## Step 11: Create Iteration 1 Tasks

[CRITICAL] Do this NOW, before invoking `/system-optimizer`. Tasks persist
across context compactions — creating them here ensures the first iteration's
checklist survives even if the skill text scrolls out of context later.

Create all 12 tasks via TaskCreate, then wire dependencies via TaskUpdate.

**Tasks to create:**

1. `Iter 1: Profile` — Profile current best mode with nsys
2. `Iter 1: Analyze + approach` — Analyze trace, identify bottleneck, select approach class
3. `Iter 1: KB search` — 2+ search_kb_deep queries, read top results IN FULL
4. `Iter 1: Prior work check` — Re-read codebase_analysis.md, check iterations.jsonl
5. `Iter 1: Hypothesis` — ONE hypothesis from profiling + KB + codebase analysis
6. `Iter 1: Implement` — ONE optimization in model_runner/optimizations/\<mode\>.py
7. `Iter 1: Correctness` — test_correctness.py (if FAIL: follow Failure Protocol, do NOT mark complete)
8. `Iter 1: Benchmark` — bench.py --compare-baseline (if regression: follow Failure Protocol)
9. `Iter 1: Judge` — Spawn sys-optimization-judge
10. `Iter 1: Profile optimized` — MANDATORY nsys profile + diff
11. `Iter 1: Commit + notes` — git commit, update state.json/iterations.jsonl/hypotheses.md
12. `Iter 1: Stop check` — Mechanical check of stop conditions

**After creating all 12, set dependencies via TaskUpdate addBlockedBy:**
```
Task 2  blockedBy: [Task 1]
Task 3  blockedBy: [Task 2]
Task 4  blockedBy: [Task 2]
Task 5  blockedBy: [Task 3, Task 4]
Task 6  blockedBy: [Task 5]
Task 7  blockedBy: [Task 6]
Task 8  blockedBy: [Task 7]
Task 9  blockedBy: [Task 8]
Task 10 blockedBy: [Task 8]
Task 11 blockedBy: [Task 9, Task 10]
Task 12 blockedBy: [Task 11]
```

Since you already profiled the baseline in Step 7, you can mark
`Iter 1: Profile` as completed immediately — the baseline trace is your
starting point.

Setup is complete. Proceed to the `/system-optimizer` skill for the
optimization loop.
