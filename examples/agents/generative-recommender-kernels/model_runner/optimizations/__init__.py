"""Optimization modules for model_runner.

Each .py file in this package (except __init__.py) is an optimization mode.
Every mode module must expose:

    def apply(model, batch, hstu_config, **kwargs) -> callable

The returned callable has signature:
    (uih_features, candidates_features) -> model_output

where model_output matches the baseline: (user_emb, item_emb, ?, mt_target_preds).
"""

from __future__ import annotations

import importlib
from pathlib import Path


def discover_modes() -> list[str]:
    """Return names of all optimization modules in this package."""
    pkg_dir = Path(__file__).parent
    return sorted(
        p.stem
        for p in pkg_dir.glob("*.py")
        if p.stem != "__init__" and not p.stem.startswith("_")
    )


def load_mode(name: str):
    """Import optimizations.<name> and return its apply() function.

    Raises ImportError if the module doesn't exist, AttributeError if
    it lacks an apply() function.
    """
    mod = importlib.import_module(f".{name}", package=__package__)
    return mod.apply
