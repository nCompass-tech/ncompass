---
name: dummy-model-setup
description: >
  Create reduced-size dummy models for profiling on a single GPU. Load when you need to
  profile a model that's too large for available hardware, create a fake/dummy model for
  benchmarking, use --load-format dummy, or reduce model layers/experts for testing.
---

# Dummy Model Setup for Single-GPU Profiling

When a model is too large for available hardware (e.g. DeepSeek V3 685B on a single H100),
you can create a reduced-size config and use vLLM's `--load-format dummy` to load random
weights. The execution pipeline (scheduling, attention, MoE routing, CUDA graphs) is real —
only the weights are fake.

## Quick start

```bash
# Create a 4-layer dummy config for DeepSeek V3
python skills/dummy-model-setup/create_dummy_config.py deepseek-ai/DeepSeek-V3 \
    --num-layers 4 --num-experts 8 --output ./dummy_models/deepseek_v3_4L

# Run with dummy weights
vllm serve ./dummy_models/deepseek_v3_4L --load-format dummy
```

## What `--load-format dummy` does

vLLM builds the full model architecture from config.json and initializes all weights with
deterministic random values (seed=1234, range [-1e-3, 1e-3]). No weight files are needed.

**Crucially, the weights are reproducible.** The dummy weight initializer uses a fixed seed
with per-parameter RNG, so running twice with the same config produces identical weights.
This means you can:
1. Profile a baseline
2. Make changes (e.g. kernel fusion, backend swap)
3. Profile again
4. Compare outputs for correctness — they should match exactly

## What to reduce

| Parameter | Impact | Risk |
|-----------|--------|------|
| `num_hidden_layers` | Massive — linear reduction in params and memory | Safe. vLLM builds fewer layers. |
| `n_routed_experts` (MoE models) | Large — experts dominate MoE param count | Safe for profiling. Routing behavior changes. |
| Hidden dimensions | Large | Risky — must stay consistent with head counts and FFN dims |

**Recommendation:** Start by reducing layers only. 4-8 layers is usually enough to see
realistic kernel behavior while fitting on one GPU.

## Script usage

```
python skills/dummy-model-setup/create_dummy_config.py <model_name> \
    --output <dir> \
    [--num-layers N] \
    [--num-experts N] \
    [--config-only]
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--num-layers N` | 4 | Number of hidden layers |
| `--num-experts N` | unchanged | Number of routed experts (MoE only) |
| `--config-only` | False | Patch existing config without re-downloading |

The script downloads `config.json` + tokenizer files from HuggingFace, patches the config,
and saves to the output directory. No model weights are downloaded.

## Examples

**DeepSeek V3 (685B MoE) on single H100:**
```bash
python skills/dummy-model-setup/create_dummy_config.py deepseek-ai/DeepSeek-V3 \
    --num-layers 4 --num-experts 8 --output ./dummy_models/deepseek_v3_small

vllm serve ./dummy_models/deepseek_v3_small --load-format dummy --moe-backend triton
```

**Qwen3.5-35B on single H100 (already fits, but want faster iteration):**
```bash
python skills/dummy-model-setup/create_dummy_config.py Qwen/Qwen3.5-35B-A3B-FP8 \
    --num-layers 4 --output ./dummy_models/qwen3.5_small

vllm serve ./dummy_models/qwen3.5_small --load-format dummy
```

**Profiling with nsys (combine with vllm-runner-profiler skill):**
```bash
ncompass profile --nsys -o traces/dummy_deepseek -- \
    vllm bench latency --model ./dummy_models/deepseek_v3_small \
        --load-format dummy --batch-size 1 --input-len 2048 --output-len 256 \
        --num-iters 1 --num-iters-warmup 1 --profile --profiler-config.profiler cuda
```

## Correctness checking across changes

Since dummy weights are deterministic (fixed seed=1234), you can verify that code changes
(fusion, backend swap, etc.) don't break correctness:

```bash
# Baseline
vllm bench latency --model ./dummy_models/my_model --load-format dummy \
    --batch-size 1 --input-len 100 --output-len 50 --num-iters 1 2>&1 | tee baseline.log

# After changes
vllm bench latency --model ./dummy_models/my_model --load-format dummy \
    --batch-size 1 --input-len 100 --output-len 50 --num-iters 1 2>&1 | tee modified.log

# Compare outputs
diff baseline.log modified.log
```

Outputs should match exactly. If they don't, the change introduced a numerical difference.
