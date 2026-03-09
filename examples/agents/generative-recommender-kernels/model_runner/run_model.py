import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict

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
from generative_recommenders.modules.dlrm_hstu import DlrmHSTU, DlrmHSTUConfig
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection, EmbeddingCollection


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
            torch.cuda.synchronize()
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
    run_profile(model, batch, args.profile_iters)

    print("Done.")


if __name__ == "__main__":
    main()
