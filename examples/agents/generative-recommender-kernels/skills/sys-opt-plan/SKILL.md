---
name: sys-opt-plan
description: >
  Synthesize recon findings into an optimization plan. Cross-references trace
  data with code barriers, assesses technique feasibility, and engages the
  developer on key design decisions before producing an agreed action plan.
---

# System Optimization — Planning

You have recon data in `.agent/notes/recon/`. Your job is to synthesize it
into an optimization plan — but NOT alone. This skill operates like plan
mode: you draft analysis, surface decision points, and produce a final plan
only after the developer has weighed in.

## Prerequisites

- `/sys-opt-recon` has completed
- `.agent/notes/recon/` contains: `entry_point.md`, `call_graph.md`,
  `trace_analysis.md`, `barriers.md`, `metadata.json`

Read all recon files before proceeding.

---

## Phase 1: Bottleneck Decomposition

Using the trace analysis and call graph, build a time budget that accounts
for the full iteration wall time. Every millisecond should be attributed
to a phase of the forward pass.

Present this to the developer:

```markdown
## Time Budget

Wall time per iteration: X.XX ms
GPU kernel time: X.XX ms (Y% of wall)

| Phase | Est. time (ms) | % of wall | Source |
|-------|----------------|-----------|--------|
| [phase from call graph] | X.XX | YY% | [trace evidence] |
| ...                     | ...  | ... | ...              |
| Unattributed overhead   | X.XX | YY% | [framework, dispatch, etc.] |

The dominant cost is [phase] at XX% of wall time.
```

If the time budget doesn't add up (gaps > 10% unattributed), say so.
Unattributed time is a finding, not an error.

## Phase 2: Technique Feasibility Assessment

For each major optimization technique, assess feasibility against the
barriers found in recon. The standard techniques to evaluate:

### CUDA Graphs
- **Requirement**: Static shapes, no CPU-GPU sync, no host-side allocation
  during capture
- **Check against barriers**: Which barriers block full-graph capture?
  Which phases could be captured if isolated?
- **Piecewise possibility**: Can the forward pass be split into
  capturable and non-capturable segments?

### torch.compile
- **Requirement**: Traceable operations, no graph breaks in critical path
- **Check against barriers**: Which barriers cause graph breaks? Which
  modules are compile-friendly if isolated?
- **Mode assessment**: Would `default`, `reduce-overhead`, or
  `max-autotune` be most appropriate given the bottleneck profile?

### Operator Fusion / Launch Reduction
- **Requirement**: Adjacent kernels with compatible data flow
- **Check against trace**: Are there clusters of small kernels that
  could be fused? What's the launch overhead contribution?

### Eager-Mode Optimization
- **Techniques**: Sync elimination, dtype precast, redundant computation
  removal, framework overhead reduction
- **Check against barriers**: Which barriers can be removed without
  changing the execution model?
- **Ceiling estimate**: How much speedup is possible from eager-mode
  improvements alone?

Write the assessment but do NOT finalize it. Present it to the developer
with decision points.

## Phase 3: Decision Points

[CRITICAL] Before writing the action plan, present the following to the
developer and wait for their input on each decision point.

### Decision Point 1: Primary Strategy

```markdown
## Decision: Primary Optimization Strategy

Based on the analysis:
- GPU utilization is XX% → [what this implies]
- The dominant cost is [phase] → [what this implies]
- Key barriers are: [list]

**Option A: [technique]**
- Expected ceiling: ~X.Xx
- Risk: [what could go wrong]
- Barriers to address: [list]

**Option B: [technique]**
- Expected ceiling: ~X.Xx
- Risk: [what could go wrong]
- Barriers to address: [list]

**Option C: Layered approach** (eager optimizations first, then [technique])
- Expected ceiling: ~X.Xx
- Risk: [complexity, diminishing returns]

Which direction should we pursue? Do you have context on any of these
barriers that would change the assessment?
```

### Decision Point 2: Barrier Resolution

For each critical barrier that blocks the primary strategy:

```markdown
## Decision: [Barrier Name]

[Barrier description from recon]

This blocks [technique] because [reason].

**Possible resolutions:**
1. [Resolution A] — [tradeoff]
2. [Resolution B] — [tradeoff]
3. Accept the barrier and work around it — [what this means for ceiling]

Do you know whether this barrier is load-bearing (must stay) or can be
worked around? Any domain knowledge about this operation?
```

### Decision Point 3: Scope and Stopping Criteria

```markdown
## Decision: Scope

Given the baseline of X.XX ms and the technique feasibility:

- **Realistic target**: X.Xx (based on [technique] ceiling minus risk)
- **Stretch target**: X.Xx (requires [barrier] resolution)
- **Diminishing returns threshold**: X% improvement per iteration

How many iterations should we budget? Any time constraints?
What speedup would you consider "good enough"?
```

## Phase 4: Compile the Action Plan

After receiving developer input on all decision points, write the final
plan to `.agent/notes/plan.md`:

```markdown
# Optimization Plan

## Context
- Baseline: X.XX ms
- GPU utilization: XX%
- Dominant cost: [from decomposition]
- Primary strategy: [from Decision 1]
- Target: X.Xx speedup

## Agreed Decisions
- Strategy: [what was decided and why]
- Barrier resolutions: [what was decided for each]
- Scope: [iterations, time, target]

## Action Sequence

### Iteration 1: [name]
- **Goal**: [specific, measurable]
- **Approach**: [technique + barrier resolution if needed]
- **Implementation sketch**: [key code changes, not full implementation]
- **Expected impact**: ~X.Xx
- **Risk**: [what could fail, and what to do if it does]
- **Success criteria**: correctness passes, speedup >= X.Xx

### Iteration 2: [name]
- **Goal**: ...
- **Depends on**: Iteration 1 [succeeding | regardless of result]
- ...

### Iteration 3: [name] (if needed)
- ...

## Fallback Plan
If the primary strategy fails:
- [What to try instead]
- [At what point to abandon and switch]

## Open Questions
- [Anything unresolved that may need revisiting]
```

## Phase 5: KB Research (Optional)

If the knowledge base is available and the plan involves techniques with
known barrier patterns, search for relevant reference material:

```
search_kb_deep: "[technique] [specific barrier pattern]"
```

Read the top results in full. If any are directly relevant to the plan,
append a "Reference Material" section to `plan.md` with file paths and
key takeaways.

---

## What NOT to do in this skill

- Do NOT start implementing — that's `/sys-opt-execute`
- Do NOT make strategic decisions without developer input
- Do NOT skip decision points even if the answer seems obvious
- Do NOT produce vague plans ("try torch.compile") — each iteration
  needs a specific implementation sketch and success criteria

When the plan is written and committed, tell the developer that planning
is complete and suggest running `/sys-opt-execute` to begin implementation.
