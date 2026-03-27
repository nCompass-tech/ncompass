#!/usr/bin/env python3
"""Correctness test: compare optimization mode outputs against baseline forward.

Examples:
  python model_runner/test_correctness.py --mode my_opt
  python model_runner/test_correctness.py --all
  python model_runner/test_correctness.py --mode my_opt --atol 1e-2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generative-recommenders"))

from run_model import (
    build_configs,
    create_model,
    generate_batch,
    load_or_cache_weights,
)

from optimizations import discover_modes, load_mode


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_outputs(
    baseline_out: tuple,
    opt_out: tuple,
    *,
    atol: float,
    rtol: float,
) -> list[dict]:
    """Compare baseline and optimized outputs, return per-tensor results."""
    names = ["user_emb", "item_emb", "hidden", "mt_target_preds"]
    results = []

    for i, name in enumerate(names):
        base = baseline_out[i]
        opt = opt_out[i]

        if base is None and opt is None:
            results.append({"name": name, "pass": True, "max_diff": 0.0, "note": "both None"})
            continue
        if base is None or opt is None:
            results.append({"name": name, "pass": False, "max_diff": float("inf"),
                            "note": f"mismatch: base={'None' if base is None else 'tensor'}, "
                                    f"opt={'None' if opt is None else 'tensor'}"})
            continue

        # Optimized output may be padded (e.g. CUDA graph). Trim to baseline size.
        if opt.shape != base.shape:
            slices = tuple(slice(0, s) for s in base.shape)
            opt = opt[slices]

        max_diff = (base - opt).abs().max().item()
        ok = torch.allclose(base, opt, atol=atol, rtol=rtol)
        results.append({"name": name, "pass": ok, "max_diff": max_diff})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    available = discover_modes()
    parser = argparse.ArgumentParser(description="Test optimization correctness vs baseline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", type=str, help=f"One of: {', '.join(available)}" if available else "No modes found")
    group.add_argument("--all", action="store_true", help="Test all discovered modes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=100_000)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--kernel", choices=["pytorch", "triton"], default="triton")
    parser.add_argument("--cache-dir", type=str,
                        default=str(Path(__file__).resolve().parent / ".model_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--notes-dir", type=str, default=None,
                        help="Auto-write last_correctness.json to this directory (default: .agent/notes/ if it exists)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    hstu_config, table_configs = build_configs(args.num_embeddings, args.max_seq_len)

    from generative_recommenders.common import HammerKernel

    model = create_model(hstu_config, table_configs, device)
    if not args.no_cache:
        load_or_cache_weights(model, args.num_embeddings, args.cache_dir, device)
    model.eval()
    kernel = HammerKernel.PYTORCH if args.kernel == "pytorch" else HammerKernel.TRITON
    model.set_hammer_kernel(kernel)

    batch = generate_batch(hstu_config, args.batch_size, device)

    # --- Baseline ---
    print("Running baseline forward...")
    with torch.no_grad():
        baseline_out = model(batch.uih_features_kjt, batch.candidates_features_kjt)
    torch.cuda.synchronize()

    # --- Test modes ---
    modes = discover_modes() if args.all else [args.mode]
    if not modes:
        print("No optimization modes found in optimizations/")
        sys.exit(1)

    all_passed = True
    notes_records: list[dict] = []

    for mode_name in modes:
        print(f"\nTesting mode: {mode_name}")
        try:
            apply_fn = load_mode(mode_name)
        except (ImportError, AttributeError) as e:
            print(f"  SKIP — failed to load: {e}")
            all_passed = False
            continue

        try:
            forward_fn = apply_fn(model, batch, hstu_config)
        except Exception as e:
            print(f"  FAIL — apply() raised: {e}")
            all_passed = False
            continue

        try:
            with torch.no_grad():
                opt_out = forward_fn(batch.uih_features_kjt, batch.candidates_features_kjt)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  FAIL — forward raised: {e}")
            all_passed = False
            continue

        results = _compare_outputs(baseline_out, opt_out, atol=args.atol, rtol=args.rtol)
        mode_passed = all(r["pass"] for r in results)
        if not mode_passed:
            all_passed = False

        for r in results:
            status = "PASS" if r["pass"] else "FAIL"
            note = f" ({r['note']})" if r.get("note") else ""
            print(f"  {r['name']:20s} {status}  max_diff={r['max_diff']:.6f}{note}")

        print(f"  {'PASSED' if mode_passed else 'FAILED'}")
        notes_records.append({
            "mode": mode_name,
            "overall": "PASS" if mode_passed else "FAIL",
            "results": [{k: v for k, v in r.items() if k != "note" or v} for r in results],
        })

    # --- Auto-capture to .agent/notes/ ---
    notes_dir = Path(args.notes_dir) if args.notes_dir else Path(".agent/notes")
    if notes_dir.is_dir() and notes_records:
        note = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall": "PASS" if all_passed else "FAIL",
            "modes": notes_records,
        }
        (notes_dir / "last_correctness.json").write_text(json.dumps(note, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
