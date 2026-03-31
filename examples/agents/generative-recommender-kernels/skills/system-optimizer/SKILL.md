---
name: system-optimizer
description: >
  Iteratively optimize the DLRM-v3 + HSTU end-to-end inference pipeline for
  system-level performance. Focuses on the optimization loop: profile, analyze,
  search, implement, test, benchmark, diff, commit.
---

# System-Level Optimizer

You are a GPU systems optimization expert. Your goal is to reduce end-to-end
iteration latency of the DLRM-v3 + HSTU inference pipeline by identifying and
fixing system-level bottlenecks.

## First-Time Setup

If `.agent/notes/state.json` does not exist, invoke `/system-optimizer-setup`
first. It handles environment setup, baseline capture, codebase analysis, and
initial profiling. Do not skip this.

If `state.json` exists, this is a resumed session — read it and the tail of
`iterations.jsonl` to recover state.

## Scope

System-level optimization ONLY: torch.compile, CUDA graphs, operator fusion,
launch overhead reduction, async CPU-GPU overlap, synchronization elimination.
Changes only in `model_runner/optimizations/`. See `/system-optimizer-reference`
for commands, schemas, and constraints.

[CRITICAL] All commands MUST use `--max-seq-len 256`.

[CRITICAL] Output must remain correct. Every optimization must pass
`test_correctness.py` before benchmarking.

---

## Optimization Loop

Each iteration follows this checklist. Every box must be checked.

### Step 1: Profile
```
□ Profile current best mode with nsys (see /system-optimizer-reference)
```

### Step 2: Analyze trace
```
□ If ncompass available: analyze_nsys_sqlite (kernel count, top kernels,
  launch overhead, sync points)
□ Identify the dominant bottleneck for THIS iteration
□ Update .agent/notes/bottleneck.md
```

### Step 3: Consult knowledge base
```
□ Search KB for techniques addressing the specific bottleneck identified
  in step 2 (use search_kb_deep for technique research, not search_kb)
□ Read the top 2-3 results in full via read_kb_file
□ If results reference code, use list_kb_adjacent to explore related files
□ If initial results are too generic, refine your query with specific
  details from the bottleneck analysis (module names, kernel types, error
  messages)
```

KB search is not one-and-done. After your initial search:
- If results are too generic, refine your query with more problem specific details
- If a result snippet looks relevant, use read_kb_file to get the full
  content and list_kb_adjacent to explore related files
- Search at least twice per iteration: once for the bottleneck class, once
  for the specific technique you plan to implement

### Step 4: Consult codebase analysis
```
□ Re-read .agent/notes/codebase_analysis.md
□ Check if the bottleneck maps to a specific finding from the analysis
  (e.g., a dtype conversion, an unused code path, a compile-friendly module)
□ Check iterations.jsonl for similar approaches already tried
```

### Step 5: Formulate hypothesis
```
□ Write ONE hypothesis informed by profiling + KB + codebase analysis
□ Update .agent/notes/hypotheses.md
```

### Step 6: Implement
```
□ Implement ONE optimization in model_runner/optimizations/<mode>.py
□ One hypothesis, one bounded code change
```

### Step 7: Test correctness
```
□ python model_runner/test_correctness.py --max-seq-len 256 --mode <mode>
□ If FAIL → go to Step 7a (do NOT skip this)
```

### Step 7a: On ANY failure — consult KB before reverting
```
□ BEFORE reverting or pivoting, spawn the sys-kb-advisor subagent:
    "My <technique> optimization failed with: <exact error or symptom>.
     The optimization module is model_runner/optimizations/<mode>.py.
     What does the KB suggest?"
□ Read the advisor's response
□ Log the KB recommendation in iterations.jsonl ("kb_recommendation" field)
□ Only THEN decide: attempt the KB-suggested fix, or revert
```

This step is MANDATORY. Do not skip it. The sys-kb-advisor subagent searches
the knowledge base with a fresh context and returns structured recommendations.
It costs ~5k tokens and takes under a minute — far cheaper than blind
trial-and-error.

### Step 8: Benchmark
```
□ python model_runner/bench.py --max-seq-len 256 --mode <mode> \
    --compare-baseline model_runner/baselines/ref.json
□ Read last_bench.json for results (do not transcribe terminal output)
□ If regression → go to Step 7a (consult KB before reverting)
```

### Step 8a: Judge optimization for overfitting [REQUIRED]

Before profiling, spawn the `sys-optimization-judge` subagent to check for
reward-hacking patterns (input caching, preprocess skipping, ignoring
arguments). This is mandatory — do not skip it.

```
□ Spawn sys-optimization-judge with:
    "Module: model_runner/optimizations/<mode>.py
     Claimed speedup: <X.Xx>"
□ Read the verdict: CLEAN, SUSPECT, or REJECT
□ If REJECT → revert immediately, the optimization is invalid
□ If SUSPECT → log the concern in iterations.jsonl, consider whether
  the flagged pattern is essential to the approach or can be removed
□ If CLEAN → proceed to Step 9
```

### Step 9: Profile optimized path + diff [REQUIRED]

This step is NOT optional. You MUST profile after every kept optimization.
Skipping this means you have no data for the next iteration's bottleneck
analysis.

```
□ Profile the optimized mode with nsys (see /system-optimizer-reference)
□ If ncompass available: analyze_nsys_diff (before trace vs after trace)
  → Quantify: kernel count change, launch overhead reduction, new hotspots
□ If ncompass available: analyze_nsys_patterns to check iteration stability
□ Record in iterations.jsonl: "profiled_after": true, "nsys_trace": "<path>"
```

### Step 10: Commit and update notes
```
□ git add model_runner/optimizations/ .agent/notes/
□ git commit with correctness/benchmark results in message
□ Append to iterations.jsonl (read last_bench.json + last_correctness.json)
□ Update state.json (iteration count, best metrics)
□ Rewrite hypotheses.md (remove tried, add new ideas from this iteration)
□ Overwrite bottleneck.md if profiling was done
```

### Step 11: Check stop conditions
```
□ Diminishing returns: last 3 iterations each <2% improvement
□ Time limit: 2 hours elapsed
□ Regression streak: 3 consecutive reverts with no success between
□ Correctness wall: 5 consecutive correctness failures
□ If any triggered → go to Wrap-Up
□ Otherwise → go to Step 1
```

---

## Wrap-Up

When any stop condition triggers:

1. Ensure the best-performing mode is current state
2. Run a final benchmark with 30 iterations:
   ```bash
   python model_runner/bench.py --max-seq-len 256 --mode <best_mode> \
     --compare-baseline model_runner/baselines/ref.json --bench-iters 30
   ```
3. Write `.agent/notes/summary.md`:
   ```markdown
   # Session Summary
   **Session ID:** <id>
   **Stop reason:** <which condition>
   **Total iterations:** <N>

   ## Results
   | Metric | Value |
   |--------|-------|
   | Baseline median | X.XX ms |
   | Best median | X.XX ms |
   | Best mode | <name> |
   | Speedup | X.XXx |

   ## Optimizations kept
   - <mode>: <description> (X.XXx)

   ## Optimizations reverted
   - <mode>: <description> — <root_cause>

   ## Remaining hypotheses
   - <from hypotheses.md>
   ```
4. Update `state.json` with `"status": "complete"` and stop reason
5. Final commit:
   ```bash
   git add .agent/notes/ model_runner/optimizations/
   git commit -m "Session complete: <best_speedup>x speedup (<stop_reason>)"
   ```
