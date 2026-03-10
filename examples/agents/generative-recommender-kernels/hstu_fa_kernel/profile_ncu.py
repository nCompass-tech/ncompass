#!/usr/bin/env python3
"""
Python entry point for NCU profiling of the from-scratch HSTU attention kernel.
Creates fixed-length inputs, does warmup, then runs the kernel under
cudaProfilerStart/Stop for clean NCU capture.

Usage (typically called via profile_ncu_runner.py):
    python hstu_fa_kernel/profile_ncu.py
    python hstu_fa_kernel/profile_ncu.py --batch-size 256 --max-seq-len 512
"""

import argparse

import torch
import torch.cuda.profiler as profiler


def main():
    parser = argparse.ArgumentParser(description="NCU profiling entry point")
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--max-seq-len", "-s", type=int, default=256)
    parser.add_argument("--num-heads", "-n", type=int, default=4)
    parser.add_argument("--head-dim", "-d", type=int, default=128)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    parser.add_argument("--softmax", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16
    num_softmax_heads = args.num_heads if args.softmax else 0

    import hstu_ai_optimized._C  # noqa: F401
    op_fn = torch.ops.hstu_ai_optimized.hstu_mha_fwd
    print(f"Profiling: hstu_ai_optimized (from-scratch)")

    print(f"Config: B={args.batch_size}, S={args.max_seq_len}, "
          f"H={args.num_heads}, D={args.head_dim}, causal={args.causal}")

    # Create fixed-length jagged inputs (all sequences = max_seq_len)
    total_tokens = args.batch_size * args.max_seq_len
    seq_offsets = torch.arange(
        0, total_tokens + 1, args.max_seq_len,
        dtype=torch.int32, device=device,
    )
    q = torch.randn(total_tokens, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, args.num_heads, args.head_dim, device=device, dtype=dtype)

    def run():
        return op_fn(
            args.max_seq_len,   # max_seq_len
            1.0,                # alpha
            q, k, v,
            seq_offsets,        # seq_offsets
            args.causal,        # causal
            None,               # num_targets
            None,               # attn_scale
            0,                  # max_attn_len
            0,                  # min_full_attn_seq_len
            0,                  # contextual_seq_len
            None,               # q_descale
            None,               # k_descale
            None,               # v_descale
            0,                  # sm_margin
            0,                  # max_q_len
            None,               # seq_offsets_q
            num_softmax_heads,  # num_softmax_heads
            False,              # training
        )

    # Warmup
    print(f"Warmup ({args.warmup} iterations)...")
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # Profiled run
    print("Running profiled iteration...")
    profiler.start()
    run()
    profiler.stop()
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
