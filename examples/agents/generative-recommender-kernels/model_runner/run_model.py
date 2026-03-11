import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch

warnings.filterwarnings("ignore", message="Logical operators.*deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="Enable tracemalloc", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "generative-recommenders"))

from generative_recommenders.common import HammerKernel
from generative_recommenders.dlrm_v3.configs import (
    get_embedding_table_config,
    get_hstu_configs,
)
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn, get_random_data
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig, SequenceEmbedding
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection


def compute_max_packed_sizes(hstu_config, batch_size):
    """Compute max possible packed tensor sizes from config."""
    contextual_features = list(hstu_config.contextual_feature_to_max_length.keys())
    max_uih_len = (hstu_config.max_seq_len
                   - hstu_config.max_num_candidates
                   - len(contextual_features))
    max_num_candidates = hstu_config.max_num_candidates

    return {
        "max_uih_len": max_uih_len,
        "max_num_candidates": max_num_candidates,
        "max_total_uih": batch_size * max_uih_len,
        "max_total_cand": batch_size * max_num_candidates,
        "batch_size": batch_size,
    }


def _pad_tensor(tensor, target_size):
    """Pad a 1D or 2D tensor to target_size along dim 0."""
    if tensor.shape[0] >= target_size:
        return tensor
    if tensor.dim() == 1:
        padded = torch.zeros(target_size, device=tensor.device, dtype=tensor.dtype)
    else:
        padded = torch.zeros(target_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
    padded[:tensor.shape[0]] = tensor
    return padded


def pad_preprocess_outputs(seq_embeddings, payload_features, max_uih_len,
                           uih_seq_lengths, max_num_candidates, num_candidates,
                           max_sizes, hstu_config):
    """Pad packed tensors from preprocess() to max theoretical sizes.

    UIH features are padded to max_total_uih, candidate features to max_total_cand.
    Features are classified using config (user vs item embedding names, merge mapping).
    Lengths/offsets tensors keep their real values (fixed shape).
    """
    max_total_uih = max_sizes["max_total_uih"]
    max_total_cand = max_sizes["max_total_cand"]

    # Build lookup sets from merge mapping — only merged features need padding.
    # Contextual features (e.g. viewer_id, dummy_contexual) have fixed sizes
    # and should NOT be padded.
    uih_merge_names = set()
    cand_merge_names = set()
    for uih_name, cand_name in hstu_config.merge_uih_candidate_feature_mapping:
        uih_merge_names.add(uih_name)
        cand_merge_names.add(cand_name)

    # Pad seq_embeddings: only features involved in the merge step
    padded_seq = {}
    for key, se in seq_embeddings.items():
        emb = se.embedding
        if key in uih_merge_names:
            padded_emb = _pad_tensor(emb, max_total_uih)
        elif key in cand_merge_names:
            padded_emb = _pad_tensor(emb, max_total_cand)
        else:
            # Fixed-size feature (contextual) — leave as-is
            padded_emb = emb
        padded_seq[key] = SequenceEmbedding(lengths=se.lengths, embedding=padded_emb)

    # Pad payload_features
    padded_pf = {}
    for key, val in payload_features.items():
        if not isinstance(val, torch.Tensor) or val.dim() < 1:
            padded_pf[key] = val
        elif "offsets" in key:
            padded_pf[key] = val
        elif key in uih_merge_names:
            padded_pf[key] = _pad_tensor(val, max_total_uih)
        elif key in cand_merge_names:
            padded_pf[key] = _pad_tensor(val, max_total_cand)
        else:
            padded_pf[key] = val

    return (padded_seq, padded_pf, max_uih_len, uih_seq_lengths,
            max_num_candidates, num_candidates)


class CUDAGraphDlrmHSTU(DlrmHSTU):
    """DlrmHSTU subclass that avoids .item() in _user_forward for CUDA graph compatibility."""

    _precomputed_total_targets: Optional[int]

    @classmethod
    def from_existing(cls, model: DlrmHSTU) -> "CUDAGraphDlrmHSTU":
        model.__class__ = cls
        model._precomputed_total_targets = None
        return model

    def _user_forward(self, max_uih_len, max_candidates, seq_embeddings,
                      payload_features, num_candidates):
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths
        source_timestamps = concat_2D_jagged(
            max_seq_len=max_uih_len + max_candidates,
            max_len_left=max_uih_len,
            offsets_left=payload_features["uih_offsets"],
            values_left=payload_features[
                self._hstu_configs.uih_action_time_feature_name
            ].unsqueeze(-1),
            max_len_right=max_candidates,
            offsets_right=payload_features["candidate_offsets"],
            values_right=payload_features[
                self._hstu_configs.candidates_querytime_feature_name
            ].unsqueeze(-1),
            kernel=self.hammer_kernel(),
        ).squeeze(-1)

        total_targets = self._precomputed_total_targets

        embedding = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].embedding
        dtype = embedding.dtype
        if (not self.is_inference) and self._bf16_training:
            embedding = embedding.to(torch.bfloat16)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=(not self.is_inference) and self._bf16_training,
        ):
            candidates_user_embeddings, _ = self._hstu_transducer(
                max_uih_len=max_uih_len,
                max_targets=max_candidates,
                total_uih_len=source_timestamps.numel() - total_targets,
                total_targets=total_targets,
                seq_embeddings=embedding,
                seq_lengths=source_lengths,
                seq_timestamps=source_timestamps,
                seq_payloads=self._construct_payload(
                    payload_features=payload_features,
                    seq_embeddings=seq_embeddings,
                ),
                num_targets=num_candidates,
            )
        return candidates_user_embeddings.to(dtype)


class CUDAGraphRunner:
    """Captures preprocess() normally, main_forward() as a CUDA graph.

    All packed tensors are padded to max theoretical sizes so tensor shapes
    are identical across batches, enabling safe CUDA graph replay.
    """

    def __init__(self, model: CUDAGraphDlrmHSTU, hstu_config, batch_size: int):
        self.model = model
        self.hstu_config = hstu_config
        self.max_sizes = compute_max_packed_sizes(hstu_config, batch_size)
        self.graph = None
        self.static_inputs = None
        self.static_output = None

    def _preprocess_and_pad(self, uih_features, candidates_features):
        """Run preprocess, pad to max sizes, set total_targets to max."""
        with torch.no_grad():
            outputs = self.model.preprocess(uih_features, candidates_features)
        (seq_embeddings, payload_features, max_uih_len,
         uih_seq_lengths, max_num_candidates, num_candidates) = outputs

        # Must use max_total_cand (not real sum) so HSTU transducer output size
        # matches padded item embeddings. During graph replay, this value is
        # frozen from capture time anyway.
        self.model._precomputed_total_targets = self.max_sizes["max_total_cand"]

        return pad_preprocess_outputs(
            seq_embeddings, payload_features, max_uih_len,
            uih_seq_lengths, max_num_candidates, num_candidates,
            self.max_sizes, self.hstu_config)

    def capture(self, uih_features, candidates_features):
        """Warmup + capture graph with max-padded tensors."""
        # Warmup on side stream (triggers triton autotuning/JIT).
        # Need fresh padded inputs each iteration because main_forward mutates seq_embeddings.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self.model.main_forward(*self._preprocess_and_pad(uih_features, candidates_features))
        torch.cuda.current_stream().wait_stream(s)

        # Fresh preprocess + pad for capture
        padded = self._preprocess_and_pad(uih_features, candidates_features)
        (seq_emb, pf, mul, usl, mnc, nc) = padded

        # Save pre-merge references: main_forward mutates seq_emb dict in-place
        # during the merge step (replacing UIH entries with merged UIH+cand).
        # We need to copy into the PRE-merge tensors during replay since those
        # are the addresses the graph reads from.
        self.static_inputs = {
            "seq_embeddings": dict(seq_emb),  # shallow copy preserves pre-merge refs
            "payload_features": dict(pf),
            "max_uih_len": mul,
            "uih_seq_lengths": usl,
            "max_num_candidates": mnc,
            "num_candidates": nc,
        }

        # Capture (this mutates seq_emb/pf dicts via main_forward's merge step)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.static_output = self.model.main_forward(*padded)
        print("  CUDA graph captured.")

    def _copy_into_static(self, padded_outputs):
        """Copy padded preprocess outputs into the graph's static buffers."""
        (seq_emb, pf, mul, usl, mnc, nc) = padded_outputs
        si = self.static_inputs

        for key in seq_emb:
            si["seq_embeddings"][key].embedding.copy_(seq_emb[key].embedding)
            si["seq_embeddings"][key].lengths.copy_(seq_emb[key].lengths)
        for key in pf:
            if isinstance(pf[key], torch.Tensor):
                si["payload_features"][key].copy_(pf[key])
        si["uih_seq_lengths"].copy_(usl)
        si["num_candidates"].copy_(nc)

    def __call__(self, uih_features, candidates_features):
        """Preprocess -> pad -> copy into static buffers -> replay graph."""
        padded = self._preprocess_and_pad(uih_features, candidates_features)
        self._copy_into_static(padded)
        self.graph.replay()
        return self.static_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile DlrmHSTU forward pass on a single GPU")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-embeddings", type=int, default=100_000,
                        help="Embedding table rows per table (default 100K, original 10M)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Override max_seq_len in config (default: use config value 16384)")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--kernel", choices=["pytorch", "triton"], default="triton",
                        help="Kernel backend: pytorch or triton (default: triton)")
    parser.add_argument("--cache-dir",
                        type=str,
                        default=str(Path(__file__).resolve().parent / ".model_cache"),
                        help="Directory to cache model weights (default: model_runner/.model_cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable model weight caching")
    parser.add_argument("--cuda-graph", action="store_true",
                        help="Use piecewise CUDA graph capture for main_forward()")
    parser.add_argument("--test-correctness", action="store_true",
                        help="Verify CUDA graph output matches baseline")
    return parser.parse_args()


def build_configs(
    num_embeddings: int,
    max_seq_len: int | None = None,
) -> tuple[DlrmHSTUConfig, Dict[str, EmbeddingConfig]]:
    hstu_config = get_hstu_configs("debug")
    table_configs = get_embedding_table_config("debug")

    if max_seq_len is not None:
        hstu_config.max_seq_len = max_seq_len

    for name, config in table_configs.items():
        config.num_embeddings = num_embeddings

    return hstu_config, table_configs


def create_model(
    hstu_config: DlrmHSTUConfig,
    table_configs: Dict[str, EmbeddingConfig],
    device: torch.device,
) -> DlrmHSTU:
    t0 = time.perf_counter()
    model = DlrmHSTU(hstu_configs=hstu_config, embedding_tables=table_configs, is_inference=False)
    print(f"  DlrmHSTU init: {time.perf_counter() - t0:.2f}s")
    t0 = time.perf_counter()
    for _, module in model.named_modules():
        if isinstance(module, (EmbeddingCollection, EmbeddingBagCollection)):
            module.to_empty(device=device)
    model = model.to(device)
    print(f"  Materialize to {device}: {time.perf_counter() - t0:.2f}s")
    return model


def load_or_cache_weights(
    model: DlrmHSTU,
    num_embeddings: int,
    cache_dir: str,
    device: torch.device,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"dlrm_hstu_ne{num_embeddings}.pt"
    t0 = time.perf_counter()
    if cache_path.exists():
        print(f"Loading cached weights from {cache_path}...")
        model.load_state_dict(torch.load(cache_path, map_location=device, weights_only=True))
        print(f"  Loaded in {time.perf_counter() - t0:.2f}s")
    else:
        print(f"Saving weights to {cache_path}...")
        torch.save(model.state_dict(), cache_path)
        print(f"  Saved in {time.perf_counter() - t0:.2f}s")


def generate_batch(hstu_config: DlrmHSTUConfig, batch_size: int, device: torch.device):
    contextual_features = list(hstu_config.contextual_feature_to_max_length.keys())
    max_uih_len = (
        hstu_config.max_seq_len
        - hstu_config.max_num_candidates
        - len(contextual_features)
    )

    print(f"Generating {batch_size} random samples (max_uih_len={max_uih_len})...")
    t0 = time.perf_counter()
    samples = [
        get_random_data(
            contexual_features=contextual_features,
            hstu_uih_keys=hstu_config.hstu_uih_feature_names,
            hstu_candidates_keys=hstu_config.hstu_candidate_feature_names,
            uih_max_seq_len=max_uih_len,
            max_num_candidates=hstu_config.max_num_candidates,
        )
        for _ in range(batch_size)
    ]
    batch = collate_fn(samples)
    batch.to(device)
    print(f"  Batch generated in {time.perf_counter() - t0:.2f}s")
    return batch


def run_warmup(model: DlrmHSTU, batch, num_iters: int) -> None:
    print(f"Running {num_iters} warmup iterations...")
    t_total = time.perf_counter()
    with torch.no_grad():
        for i in range(num_iters):
            model(batch.uih_features_kjt, batch.candidates_features_kjt)
            torch.cuda.synchronize()
    print(f"  Warmup total: {time.perf_counter() - t_total:.2f}s")


def run_profile(model: DlrmHSTU, batch, num_iters: int) -> None:
    print(f"Running {num_iters} profiled iterations...")
    torch.cuda.cudart().cudaProfilerStart()
    t_total = time.perf_counter()
    with torch.no_grad():
        for i in range(num_iters):
            t0 = time.perf_counter()
            model(batch.uih_features_kjt, batch.candidates_features_kjt)
            print(f"  profile iter {i+1}/{num_iters}: {time.perf_counter() - t0:.2f}s")
    torch.cuda.cudart().cudaProfilerStop()
    print(f"  Profile total: {time.perf_counter() - t_total:.2f}s")


def test_correctness(model, batch, hstu_config):
    """Compare baseline forward vs CUDA graph forward for correctness."""
    device = next(model.parameters()).device

    # 1. Baseline forward
    with torch.no_grad():
        baseline_output = model(batch.uih_features_kjt, batch.candidates_features_kjt)
    torch.cuda.synchronize()

    # 2. CUDA graph forward
    graph_model = CUDAGraphDlrmHSTU.from_existing(model)
    runner = CUDAGraphRunner(graph_model, hstu_config, batch_size=batch.uih_features_kjt.stride())
    runner.capture(batch.uih_features_kjt, batch.candidates_features_kjt)
    graph_output = runner(batch.uih_features_kjt, batch.candidates_features_kjt)
    torch.cuda.synchronize()

    # 3. Compare outputs
    baseline_user_emb, baseline_item_emb = baseline_output[0], baseline_output[1]
    graph_user_emb, graph_item_emb = graph_output[0], graph_output[1]

    # Trim graph output to baseline size (graph output is max-padded)
    graph_user_emb = graph_user_emb[:baseline_user_emb.shape[0]]
    graph_item_emb = graph_item_emb[:baseline_item_emb.shape[0]]

    user_ok = torch.allclose(baseline_user_emb, graph_user_emb, atol=1e-3, rtol=1e-3)
    item_ok = torch.allclose(baseline_item_emb, graph_item_emb, atol=1e-3, rtol=1e-3)

    # Also compare mt_target_preds (index 3) — shape is (num_tasks, L)
    if baseline_output[3] is not None:
        L = baseline_output[3].shape[1]
        preds_ok = torch.allclose(
            baseline_output[3], graph_output[3][:, :L],
            atol=1e-3, rtol=1e-3)
    else:
        preds_ok = True

    if user_ok and item_ok and preds_ok:
        print("CORRECTNESS TEST PASSED")
    else:
        print(f"CORRECTNESS TEST FAILED: user_emb={user_ok}, item_emb={item_ok}, preds={preds_ok}")
        print(f"  user_emb max diff: {(baseline_user_emb - graph_user_emb).abs().max().item():.6f}")
        print(f"  item_emb max diff: {(baseline_item_emb - graph_item_emb).abs().max().item():.6f}")


def run_profile_with_runner(runner: CUDAGraphRunner, batch, num_iters: int) -> None:
    print(f"Running {num_iters} profiled iterations (CUDA graph replay)...")
    torch.cuda.cudart().cudaProfilerStart()
    t_total = time.perf_counter()
    with torch.no_grad():
        for i in range(num_iters):
            t0 = time.perf_counter()
            runner(batch.uih_features_kjt, batch.candidates_features_kjt)
            print(f"  profile iter {i+1}/{num_iters}: {time.perf_counter() - t0:.2f}s")
    torch.cuda.cudart().cudaProfilerStop()
    print(f"  Profile total: {time.perf_counter() - t_total:.2f}s")


def main():
    args = parse_args()
    device = torch.device("cuda:0")

    hstu_config, table_configs = build_configs(args.num_embeddings, args.max_seq_len)
    print(f"Config: batch_size={args.batch_size}, num_embeddings={args.num_embeddings}, "
          f"max_seq_len={hstu_config.max_seq_len}, max_num_candidates={hstu_config.max_num_candidates}")

    print("Creating model...")
    model = create_model(hstu_config, table_configs, device)

    if not args.no_cache:
        load_or_cache_weights(model, args.num_embeddings, args.cache_dir, device)

    model.eval()

    kernel = HammerKernel.PYTORCH if args.kernel == "pytorch" else HammerKernel.TRITON
    model.set_hammer_kernel(kernel)
    print(f"Kernel backend: {args.kernel}")
    print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch = generate_batch(hstu_config, args.batch_size, device)
    print(f"Batch ready. UIH stride={batch.uih_features_kjt.stride()}, "
          f"candidates stride={batch.candidates_features_kjt.stride()}")

    run_warmup(model, batch, args.warmup_iters)

    if args.test_correctness:
        test_correctness(model, batch, hstu_config)
    elif args.cuda_graph:
        print("Setting up CUDA graph capture...")
        model = CUDAGraphDlrmHSTU.from_existing(model)
        runner = CUDAGraphRunner(model, hstu_config, batch_size=args.batch_size)
        runner.capture(batch.uih_features_kjt, batch.candidates_features_kjt)
        run_profile_with_runner(runner, batch, args.profile_iters)
    else:
        run_profile(model, batch, args.profile_iters)

    print("Done.")


if __name__ == "__main__":
    main()
