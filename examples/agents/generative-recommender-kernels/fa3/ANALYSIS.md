# Generative Recommenders — Repository Analysis

## Overview

This repo is **Meta's Generative Recommenders** codebase — the implementation behind the ICML'24 paper *"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"*. It reformulates classical deep learning recommendation models (DLRMs) as generative models, using the **HSTU** (Hierarchical Sequential Transduction Unit) architecture.

---

## 1) Model File Location & Running on Hopper

**Model definition:**
`generative_recommenders/research/modeling/sequential/hstu.py`

This defines the HSTU architecture — a sequential attention-based model with relative positional/temporal biases, configurable blocks, heads, and head dimensions.

**To run on your Hopper GPU (H100):**

```bash
# 1. Install dependencies
pip3 install -r requirements.txt   # Requires CUDA 12.3+, tested with 12.4

# 2. Download and preprocess data
mkdir -p tmp/ && python3 preprocess_public_data.py

# 3. Train (single GPU)
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
  --master_port=12345
```

For the production DLRM-v3 variant (multi-GPU):
```bash
# Training (4 GPUs)
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/train/train_ranker.py --dataset debug --mode train

# Inference (4 GPUs)
LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 generative_recommenders/dlrm_v3/inference/main.py --dataset debug
```

The CUDA kernels auto-detect Hopper via `is_sm90()` in `generative_recommenders/ops/utils.py` and route to SM90-optimized code paths. No special flags needed — just having an H100 is sufficient.

---

## 2) Is FA3 Being Used?

**Yes.** The README explicitly states (line 105):

> `ops/cpp/hstu_attention` contains the attention implementation based on FlashAttention V3 with state-of-the-art efficiency on H100 GPUs.

The full FA3 CUDA implementation lives at:
`generative_recommenders/ops/cpp/hstu_attention/`

Key files:
- `flash_fwd_kernel_sm90.h` — forward pass kernel (Hopper-specific)
- `flash_bwd_kernel_sm90.h` — backward pass kernel
- `mainloop_fwd_sm90_tma_gmma_ws.h` — uses Hopper's TMA + GMMA hardware
- `mainloop_bwd_sm90_tma_gmma_ws.h` — backward mainloop
- `flash_api.cpp` — C++ torch binding (`torch.ops.hstu.hstu_mha`)

The Python dispatch chain is:
```
hstu_attention.py (selects kernel backend)
  -> HammerKernel.CUDA -> cuda_hstu_attention.py
    -> torch.ops.hstu.hstu_mha()  (the FA3 CUDA kernel)
```

There are also fallback paths: `TRITON` (Triton kernels), `PYTORCH` (reference implementation).

---

## 3) Why Is FA3 Custom vs. Standard?

This is a **fork of FA3** (attributed to Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao — the original FA3 authors), **not** the standard `flash-attn` pip package. The customizations fall into two categories: masking patterns and structural kernel changes.

### a) Jagged/variable-length sequence support
Standard FA3 operates on fixed-size padded batches. HSTU uses **jagged tensors** — variable-length user histories packed without padding. The custom kernel natively takes `seq_offsets` (ragged batch offsets), `num_targets`, `contextual_seq_len`, and `max_attn_len` — none of which exist in standard FA3. See `flash.h:84-92`:
```cpp
int* __restrict__ seq_offsets;
int* __restrict__ num_targets;
float* __restrict__ attn_scale;
int max_attn_len, contextual_seq_len, min_full_attn_seq_len;
```

### b) Hopper pipeline performance fix
`sm90_pipeline_no_cluster.h:31-35` contains a **performance regression fix** for the CUTLASS TMA pipeline:
> "As of Cutlass v3.6.0, if size(ClusterShape) == 1, PipelineTmaAsync has all threads signaling the barrier during consumer_release. This causes a perf regression in FA3 forward pass (especially hdim 128 causal). We instead reimplement the version of PipelineTmaAsync before v3.6.0 where only 1 out of 128 threads signals the barrier."

### c) FP8 quantization with per-head descaling
The kernel supports FP8 (e4m3) inputs with `q_descale`, `k_descale`, `v_descale` tensors for mixed-precision inference — tailored for recommendation workloads. See `flash.h:61-69`.

### d) Group/recursive batch semantics
The params struct includes `num_groups` and `batch_size_per_group` (`flash.h:75`) for grouped batch processing, specific to production recommendation serving patterns.

---

## 4) HSTU-Specific Masking Patterns — Detailed Breakdown

The custom masking lives in **`mask.h`** (the `Mask::apply` template function, lines 58–393) and is wired into the forward mainloop at **`mainloop_fwd_sm90_tma_gmma_ws.h`** lines 1348–1680. The `Mask` struct has **6 compile-time boolean template parameters** controlling which masks are active:

```cpp
// mask.h:58-66
template <
    bool Seqlenq_mask,    // mask out-of-bounds query positions
    bool Seqlenk_mask,    // mask out-of-bounds key positions
    bool Causal_mask,     // standard causal (q can only attend to k <= q)
    bool Local_mask,      // sliding window with max_attn_len
    bool Contexual_mask,  // contextual prefix masking
    bool Target_mask,     // diagonal target-only masking
    bool Cross,           // cross-attention variant
    bool Softmax>         // -inf vs 0 masking value
```

**Standard FA3** only has the equivalent of `Causal_mask` (and basic `Seqlenq/k_mask` for padding). Here are the **4 additional mask types** this custom kernel adds:

### Mask 1: Local window attention (`Local_mask`) — `mask.h:281-390`

When `Local_mask=true`, the kernel enforces a **sliding window** where each query position `i` can only attend to keys in the range `[i - max_attn_len, i]`. This is implemented via a left boundary check:

```cpp
// mask.h:282-284
int const local_row_offset_left = causal_row_offset - 1 - max_attn_len;
int const col_limit_sink = 0 - n_block * BlockN - thread_kdim_offset;
```

At `mask.h:326-337`, for rows in the "user interaction history" (UIH) region (`row_idx < max_uih_len - min_full_attn_seq_len`), keys to the left of the window are zeroed out. The `min_full_attn_seq_len` parameter allows the last N positions to use **full** (non-local) attention — the idea being that the most recent interactions deserve broader context.

The mainloop orchestrates this by splitting the KV loop into 3 regions (`mainloop_fwd_sm90_tma_gmma_ws.h:1576-1632`):
1. **Causal boundary tiles** (lines 1580-1597): tiles near the diagonal get `Causal+Local` mask
2. **Interior tiles** (lines 1610-1613): fully within the window, no masking needed (`no_mask_fn`)
3. **Left boundary tiles** (lines 1616-1631): tiles at the left edge of the window get `Local`-only mask

Standard FA3 has no concept of a local left boundary or `min_full_attn_seq_len` override.

### Mask 2: Contextual masking (`Contexual_mask`) — `mask.h:225-240`, `mask.h:292-317`

This handles a **contextual prefix** of length `contextual_seq_len` — a set of "context" tokens that sit before the user's interaction history. The rule is:

- **Context rows** (query position < `contextual_seq_len`): these can only attend to other context positions and NOT to UIH positions. Enforced at `mask.h:225-240`:
  ```cpp
  // mask.h:225-227 (inside Causal_mask branch)
  if constexpr (Contexual_mask) {
      if (row_idx < contextual_seq_len - m_block * BlockM - thread_qdim_offset) {
          // Only allow attending within contextual region (mask out UIH keys)
          for (int n ...) {
              if (t0_col_idx >= uihlen_k_limit) { tSrS_rowcol(m, n) = 0.0f; }
          }
          continue;  // skip the causal logic for this row
  ```

- **UIH rows with local attention**: when combined with `Local_mask`, the contextual prefix acts as a **sink** — keys in the context region are always visible even if they'd fall outside the local window. See `mask.h:312-317`:
  ```cpp
  // mask.h:312-317
  if constexpr (Contexual_mask) {
      if (col_limit_left + n_block * BlockN + thread_kdim_offset < contextual_seq_len) {
          col_limit_left = 0;  // allow attending to all context tokens
      }
  }
  ```

The mainloop has a dedicated loop section for contextual tiles at `mainloop_fwd_sm90_tma_gmma_ws.h:1658-1680`.

Standard FA3 has zero concept of a contextual prefix region.

### Mask 3: Target masking (`Target_mask`) — `mask.h:184-199`, `mask.h:259-279`, `mask.h:347-388`

This is for the **recommendation target items** at the end of the sequence. The sequence is split into:
- **UIH region** (positions `0` to `uihlen_q - 1`): user interaction history
- **Target region** (positions `uihlen_q` to `seqlen - 1`): candidate items being scored

The target mask enforces that **target items can only attend to themselves** (diagonal attention) — each target only sees its own position, not other targets. At `mask.h:268-269`:
```cpp
bool const target_cond = (row_idx != col_idx) &&
    (row_idx >= max_uih_len) && (col_idx >= max_uih_len);
```

This means: if both row and col are in the target region AND they're not the same position, mask it out. Combined with the causal mask for the UIH region, each target sees [all UIH positions it's allowed + only itself from the target region].

The mainloop handles this with a separate target GEMM loop at `mainloop_fwd_sm90_tma_gmma_ws.h:1633-1656`.

Standard FA3 has no concept of UIH/target sequence splitting.

### Mask 4: Composite masking in the mainloop — `mainloop_fwd_sm90_tma_gmma_ws.h:1416-1459`

The **first tile** (containing the diagonal) applies masks **conditionally based on which region the M-block falls in**:

```
if m_block is entirely in UIH:
    apply(Causal, Local, Contexual, Target=false)       // line 1428-1436
elif m_block straddles UIH/target boundary:
    apply(Causal, Local, Contexual, Target=Has_targets)  // line 1440-1448
elif m_block is entirely in target region:
    apply(Causal=false, Local=false, Contexual, Target=Has_targets)  // line 1450-1458
```

### How `mask.apply()` works internally

`apply()` uses **compile-time `if constexpr` branching** to generate a **single fused pass** that combines all active masks per-element in one traversal:

1. `Seqlenq_mask` is checked first independently (`mask.h:117-131`), zeroing entire rows beyond the query sequence length.
2. A compile-time `if/else` tree selects the **one active masking regime** (enforced by `static_assert(!(Causal_mask && Local_mask))` at line 74 — Causal and Local are mutually exclusive):
   - `Cross` → cross-attention path (lines 132-177)
   - `!Causal && !Local` → flat path: just Seqlenk or Target column masking (lines 179-214)
   - `Causal` → causal path (lines 217-280)
   - `Local` → local path (lines 281-390)
3. Within the chosen branch, the other active masks (Contextual, Target) are fused per-element with zero runtime branching overhead for disabled masks.

### Summary: Standard FA3 vs. Custom HSTU FA3 masking

| Feature | Standard FA3 | Custom HSTU FA3 |
|---------|-------------|-----------------|
| Causal mask | Yes | Yes (`mask.h:217-279`) |
| Sequence length masking | Yes (padding) | Yes, jagged (`seqlen.h`, `mask.h:117-131`) |
| Sliding window (local) | No* | Yes, with `min_full_attn_seq_len` override (`mask.h:281-390`) |
| Contextual prefix | No | Yes, bidirectional prefix + sink tokens (`mask.h:225-240, 292-317`) |
| Target diagonal mask | No | Yes, per-item scoring (`mask.h:184-199, 259-279`) |
| Per-tile mask dispatch | Single branch | 3-way UIH/boundary/target dispatch (`mainloop:1416-1459`) |
| Mainloop loop splitting | 1 loop | Up to 5 separate loops: causal, interior, local-left, target, contextual (`mainloop:1544-1680`) |

*Standard FA3 v2.5+ added sliding window, but without contextual sinks or `min_full_attn_seq_len`.

---

## 5) Structural Kernel Differences Beyond Masking

The customizations go far beyond masking. The relationship to stock FA3 is: they took the FA3 **skeleton** (TMA+GMMA warpgroup-specialized architecture) and replaced the **flesh** (what computation happens inside the mainloop, how tiles are scheduled, how scores are transformed).

### SiLU replaces softmax as the default attention activation

Standard FA3's entire inner loop is built around **online softmax** (the Milakov-Dao trick: track running max, rescale, exponentiate). HSTU doesn't use softmax by default — it uses **SiLU (Sigmoid Linear Unit)**.

At `mainloop_fwd_sm90_tma_gmma_ws.h:1394-1412`, after the QK GEMM:

```cpp
hstu::gemm</*zero_init=*/true>(tiled_mma0, tSrQ, tSrK(...), tSrS);
for (int mi ...) {
    for (int ni ...) {
        tSrS_rowcol(mi, ni) = silu_scale_op(tSrS_rowcol(mi, ni) * params.alpha, scale);
    }
}
```

Where `SiluScaleOp` at `utils.h:120-127` is:
```cpp
T operator()(T const& t, T const& scale) {
    float t2 = t / 2;
    return t2 * (1 + cutlass::fast_tanh(t2)) * scale;
}
```

The kernel has two completely different MMA paths:
- `mma()` — **SiLU path** (no running max, no rescaling, just SiLU + accumulate)
- `mma_softmax()` — **softmax path** (online softmax with rescaling, like standard FA3)

Standard FA3 only has the softmax path. There is no way to bolt SiLU onto stock FA3 without rewriting the mainloop.

### Per-query attention scaling (not a scalar)

Standard FA3 uses a single scalar `1/sqrt(d)` for all positions. This kernel uses a **per-query-position scale tensor** (`params.attn_scale`), loaded per row at `mainloop_fwd_sm90_tma_gmma_ws.h:1400-1407`:

```cpp
if (!params.scalar_scale) {
    int q_index = qdim_offset + int(get<Qdim>(t0ScS_rowcol(mi, _0{})));
    scale = params.attn_scale[q_index];  // per-position from global memory
}
```

### Jagged tensor TMA addressing

Standard FA3 assumes fixed batch x seqlen x heads x dim layout. The TMA descriptors here are set up for **jagged/packed** tensors where sequences are concatenated (total_seqlen x heads x dim). The tile scheduler must compute per-batch offsets from `seq_offsets`.

### Custom tile schedulers

The `VarlenDynamicPersistentTileScheduler` (`tile_scheduler.h:370-614`) uses **warp-level prefix sums and ballot instructions** to efficiently map a flat tile index to the correct (batch, head, block) for variable-length sequences:

```cpp
// tile_scheduler.h:440-450
auto prefix_sum = [](int val) {
    for (int i = 1; i < NumThreadsPerWarp; i <<= 1) {
        int32_t partial_sum = __shfl_up_sync(0xffffffff, val, i);
        if (lane >= i) val += partial_sum;
    }
    return val;
};
```

Standard FA3's tile scheduler is a simple grid-stride loop over `(batch, head, m_block)`. This scheduler does dynamic work-stealing with L2-aware swizzling for jagged sequences.

### Mainloop loop splitting affects producer loads too

The producer (TMA load) side mirrors the consumer (MMA) side's loop splitting at `mainloop_fwd_sm90_tma_gmma_ws.h:1043-1118`. The producer must skip tiles that won't be needed and load target tiles from a different KV range.

### Summary: Same vs. Different from stock FA3

| Aspect | Same as stock FA3? |
|--------|-------------------|
| CUTLASS TMA/GMMA primitives | Yes |
| Shared memory layout patterns | Yes |
| Warpgroup-specialized producer/consumer structure | Yes |
| Pipeline staging | Yes |
| Attention activation function | **No** — SiLU vs softmax |
| Attention score scaling | **No** — per-position vs scalar |
| Tensor addressing | **No** — jagged vs padded |
| Tile scheduling | **No** — varlen dynamic vs fixed grid |
| Mainloop iteration pattern | **No** — 5-way loop split vs 1 loop |
| Producer load pattern | **No** — mirrors consumer splits |
