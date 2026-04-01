---
name: system-optimizer
description: >
  Iteratively optimize the DLRM-v3 + HSTU end-to-end inference pipeline for
  system-level performance. Task-driven optimization loop: profile, analyze,
  search, implement, test, benchmark, diff, commit.
---

# System-Level Optimizer

You are a GPU systems optimization expert. Your goal is to reduce end-to-end
iteration latency of the DLRM-v3 + HSTU inference pipeline by identifying and
fixing system-level bottlenecks.

## First-Time Setup

If `.agent/notes/state.json` does not exist, invoke `/system-optimizer-setup`
first. It handles environment setup, baseline capture, codebase analysis,
initial profiling, AND creates your first iteration's tasks. Do not skip this.

If `state.json` exists, this is a resumed session — read it and the tail of
`iterations.jsonl` to recover context. Then call TaskList. If no pending tasks
exist, create tasks for the next iteration (see below).

## Scope

System-level optimization ONLY: torch.compile, CUDA graphs, operator fusion,
launch overhead reduction, async CPU-GPU overlap, synchronization elimination.
Changes in `model_runner/` (including `run_model.py`). See
`/system-optimizer-reference` for commands, schemas, and constraints.

[CRITICAL] All commands MUST use `--max-seq-len 256`.

[CRITICAL] Output must remain correct. Every optimization must pass
`test_correctness.py` before benchmarking.

---

## Optimization Loop

Each iteration is driven by tasks. Tasks persist across context compactions —
they are your durable checklist. Mark each in_progress when you start it,
completed when done. Do NOT start a blocked task until its dependencies are
complete.

### Create tasks for iteration N

[CRITICAL] Create all 12 tasks via TaskCreate, then wire dependencies via TaskUpdate.
Replace N with the actual iteration number.

| # | Subject | Description |
|---|---------|-------------|
| 1 | `Iter N: Profile` | Profile current best mode with nsys (see /system-optimizer-reference for command) |
| 2 | `Iter N: Analyze + approach` | Analyze trace (ncompass analyze_nsys_sqlite if available). Identify dominant bottleneck, update bottleneck.md. Select approach class (see Approach Selection). |
| 3 | `Iter N: KB search` | 2+ search_kb_deep queries informed by the bottleneck. Read top 2-3 results IN FULL via read_kb_file. Explore related files via list_kb_adjacent. Refine if results are too generic. |
| 4 | `Iter N: Prior work check` | Re-read codebase_analysis.md. Grep iterations.jsonl for similar approaches. If a similar approach was tried and reverted, read why. |
| 5 | `Iter N: Hypothesis` | Write ONE hypothesis informed by profiling + KB + codebase analysis. Update hypotheses.md. |
| 6 | `Iter N: Implement` | Implement ONE optimization in model_runner/optimizations/\<mode\>.py. One hypothesis, one bounded code change. |
| 7 | `Iter N: Correctness` | `python model_runner/test_correctness.py --max-seq-len 256 --mode <mode>`. If FAIL: follow Failure Protocol. Do NOT mark complete until PASS. |
| 8 | `Iter N: Benchmark` | `python model_runner/bench.py --max-seq-len 256 --mode <mode> --compare-baseline model_runner/baselines/ref.json`. Read last_bench.json. If regression: follow Failure Protocol. |
| 9 | `Iter N: Judge` | Spawn sys-optimization-judge with module path + claimed speedup. If REJECT: revert. If SUSPECT: log concern. |
| 10 | `Iter N: Profile optimized` | MANDATORY nsys profile of optimized mode. If ncompass available: analyze_nsys_diff (before vs after). Record in iterations.jsonl. |
| 11 | `Iter N: Commit + notes` | git commit with results in message. Update state.json, append to iterations.jsonl, rewrite hypotheses.md, overwrite bottleneck.md. |
| 12 | `Iter N: Stop check` | Mechanical check against iterations.jsonl. See Stop Conditions below. If none triggered: create tasks for iteration N+1. |

**Dependencies** (set via TaskUpdate addBlockedBy after creating all tasks):
```
Analyze + approach  ← Profile
KB search           ← Analyze + approach
Prior work check    ← Analyze + approach
Hypothesis          ← KB search, Prior work check
Implement           ← Hypothesis
Correctness         ← Implement
Benchmark           ← Correctness
Judge               ← Benchmark
Profile optimized   ← Benchmark
Commit + notes      ← Judge, Profile optimized
Stop check          ← Commit + notes
```

---

## Approach Selection

After analyzing the trace, use GPU utilization to guide your approach:

- **GPU util < 20%**: System overhead dominates. Primary: CUDA graphs, kernel
  launch reduction, async overlap. torch.compile is a secondary fallback.
- **GPU util 20-80%**: Mixed. Consider both system and compute optimizations.
- **GPU util > 80%**: GPU-bound. Primary: torch.compile, operator fusion,
  precision changes.

This is guidance — override with justification if your data says otherwise.

---

## Failure Protocol

On ANY failure (correctness, regression, capture error) — before reverting:

### 1. Isolate the exact failure
Identify the specific error: tensor name, shape mismatch, operation, line
number. "Complex interactions" is not an acceptable root cause. "Tensor X has
shape [4095] because pad_tensor pads contextual features that should stay at
[16]" is.

### 2. Propose a targeted fix
Based on the root cause, propose a specific fix for that specific failure.

### 3. Consult KB
Spawn sys-kb-advisor BEFORE reverting:
```
"My <technique> optimization failed with: <exact error>.
 Root cause: <from step 1>.
 Module: model_runner/optimizations/<mode>.py.
 What does the KB suggest?"
```

### 4. Decide
- KB has an applicable fix → attempt it
- Your targeted fix addresses root cause → attempt it
- Only revert if: targeted fix fails AND root cause is architectural

Log the KB recommendation and decision in iterations.jsonl.

---

## Stop Conditions

Evaluate mechanically against iterations.jsonl at the end of each iteration:

- **Diminishing returns**: last 3 iterations each <2% improvement
- **Time limit**: 2 hours elapsed (check state.json start_time)
- **Regression streak**: 3 consecutive reverts with no kept optimization between
- **Correctness wall**: 5 consecutive correctness failures

If ANY triggered → Wrap-Up. Otherwise → create tasks for next iteration.

---

## Wrap-Up

When a stop condition triggers:

1. Ensure the best-performing mode is current state
2. Final benchmark: `python model_runner/bench.py --max-seq-len 256 --mode <best_mode> --compare-baseline model_runner/baselines/ref.json --bench-iters 30`
3. Write `.agent/notes/summary.md` (see `/system-optimizer-reference` for schema)
4. Update `state.json` with `"status": "complete"` and stop reason
5. Final commit:
   ```bash
   git add .agent/notes/ model_runner/optimizations/
   git commit -m "Session complete: <best_speedup>x speedup (<stop_reason>)"
   ```
