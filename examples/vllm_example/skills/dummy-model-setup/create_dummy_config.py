#!/usr/bin/env python3
"""Create a reduced-size model config for dummy-weight profiling.

Downloads config.json and tokenizer files from a HuggingFace model, patches
the config to reduce the model size (fewer layers, fewer experts), and saves
to a local directory. Use with ``vllm serve <local_dir> --load-format dummy``.

Usage:
    python create_dummy_config.py deepseek-ai/DeepSeek-V3 \
        --num-layers 4 --num-experts 8 --output ./dummy_deepseek_v3

    python create_dummy_config.py Qwen/Qwen3.5-35B-A3B-FP8 \
        --num-layers 4 --output ./dummy_qwen3.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Fields to try reducing, in order of impact
LAYER_KEYS = ["num_hidden_layers", "n_layer", "num_layers"]
EXPERT_KEYS = ["n_routed_experts", "num_local_experts", "num_experts"]


def patch_config(config: dict, num_layers: int | None, num_experts: int | None) -> dict:
    """Patch a HuggingFace config dict to reduce model size."""
    patched = config.copy()

    if num_layers is not None:
        for key in LAYER_KEYS:
            if key in patched:
                original = patched[key]
                patched[key] = num_layers
                print(f"  {key}: {original} -> {num_layers}")
                break
        else:
            print(f"  Warning: no layer count key found in config ({LAYER_KEYS})")

    if num_experts is not None:
        for key in EXPERT_KEYS:
            if key in patched:
                original = patched[key]
                patched[key] = num_experts
                print(f"  {key}: {original} -> {num_experts}")
                break
        else:
            print(f"  Warning: no expert count key found in config ({EXPERT_KEYS})")

    return patched


def download_tokenizer_and_config(model_name: str, output_dir: Path) -> dict:
    """Download config.json and tokenizer files from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        raise SystemExit(1)

    print(f"Downloading config + tokenizer from {model_name}...")
    snapshot_download(
        model_name,
        local_dir=str(output_dir),
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "generation_config.json",
        ],
    )

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {output_dir}")

    return json.loads(config_path.read_text())


def main():
    parser = argparse.ArgumentParser(
        description="Create a reduced-size model config for dummy-weight profiling.",
    )
    parser.add_argument("model", help="HuggingFace model name (e.g. deepseek-ai/DeepSeek-V3)")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for the dummy config")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of hidden layers (default: 4)")
    parser.add_argument("--num-experts", type=int, default=None,
                        help="Number of routed experts (default: unchanged)")
    parser.add_argument("--config-only", action="store_true",
                        help="Only patch an existing config.json (skip download)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.config_only:
        config_path = args.output / "config.json"
        if not config_path.exists():
            print(f"Error: {config_path} not found. Run without --config-only first.")
            raise SystemExit(1)
        config = json.loads(config_path.read_text())
    else:
        config = download_tokenizer_and_config(args.model, args.output)

    print("Patching config...")
    patched = patch_config(config, args.num_layers, args.num_experts)

    config_path = args.output / "config.json"
    config_path.write_text(json.dumps(patched, indent=2) + "\n")
    print(f"\nDummy config saved to {args.output}/")
    print(f"\nRun with:")
    print(f"  vllm serve {args.output} --load-format dummy")


if __name__ == "__main__":
    main()
