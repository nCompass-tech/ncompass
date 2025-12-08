import os
import json
from pathlib import Path

def _log_error(cache_dir: str | None, message: str) -> None:
    """Log error to file in cache dir if available."""
    if cache_dir:
        log_path = Path(cache_dir) / ".ncompass_init.log"
        try:
            with open(log_path, "a") as f:
                f.write(f"[PID={os.getpid()}] {message}\n")
        except Exception:
            pass  # Silent fail if we can't write log

cache_dir = os.environ.get("NCOMPASS_CACHE_DIR")
profiler_type = os.environ.get("NCOMPASS_PROFILER_TYPE")

# Both must be set or both must be unset
if bool(cache_dir) != bool(profiler_type):
    missing = "NCOMPASS_PROFILER_TYPE" if cache_dir else "NCOMPASS_CACHE_DIR"
    present = "NCOMPASS_CACHE_DIR" if cache_dir else "NCOMPASS_PROFILER_TYPE"
    msg = f"Both NCOMPASS_CACHE_DIR and NCOMPASS_PROFILER_TYPE must be set. " \
          f"Found {present} but missing {missing}. Unset {present} to disable profiler."
    _log_error(cache_dir, msg)
    raise RuntimeError(msg)

if cache_dir and profiler_type:
    if profiler_type not in ("NVTX", "Torch"):
        msg = f"NCOMPASS_PROFILER_TYPE must be 'NVTX' or 'Torch', got '{profiler_type}'"
        _log_error(cache_dir, msg)
        raise ValueError(msg)
    
    config_path = Path(cache_dir) / f".cache/ncompass/profiles/.default/{profiler_type}/current/config.json"
    
    if not config_path.exists():
        msg = f"Config file not found: {config_path}. " \
              f"Unset NCOMPASS_CACHE_DIR and NCOMPASS_PROFILER_TYPE to disable profiler."
        _log_error(cache_dir, msg)
        raise FileNotFoundError(msg)
    
    try:
        from ncompass.trace.core.rewrite import enable_rewrites
        from ncompass.trace.core.pydantic import RewriteConfig
        
        with open(config_path) as f:
            cfg = json.load(f)
        enable_rewrites(config=RewriteConfig.from_dict(cfg))
    except Exception as e:
        _log_error(cache_dir, f"Failed to enable rewrites: {e}")
        raise
