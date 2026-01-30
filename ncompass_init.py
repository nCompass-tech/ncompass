import os
import json
from pathlib import Path
from typing import List

# Valid profiler types (actual directory names)
# "NCU" is a special type that requires NCOMPASS_TRACE_NAME to be set
# When NCU is used, it also enables NVTX rewrites for NVTX-based filtering
VALID_PROFILER_TYPES = ("NVTX", "Torch", "CudaProfiler", "NCU")

# Aliases that expand to multiple profiler types
PROFILER_ALIASES = {
    "NSYS": ["NVTX", "CudaProfiler"]
}


def _log_error(cache_dir: str | None, message: str) -> None:
    """Log error to file in cache dir if available."""
    if cache_dir:
        log_path = Path(cache_dir) / ".ncompass_init.log"
        try:
            with open(log_path, "a") as f:
                f.write(f"[PID={os.getpid()}] {message}\n")
        except Exception:
            pass  # Silent fail if we can't write log


def _parse_profiler_types(profiler_type_str: str) -> List[str]:
    """
    Parse profiler type string into list of profiler types.

    Supports:
    - Single type: "NVTX"
    - Comma-separated: "NVTX,CudaProfiler"
    - Alias: "NSYS" -> ["NVTX", "CudaProfiler"]
    """
    profiler_type_str = profiler_type_str.strip()

    # Check if it's an alias
    if profiler_type_str in PROFILER_ALIASES:
        return PROFILER_ALIASES[profiler_type_str]

    # Parse comma-separated list
    types = [t.strip() for t in profiler_type_str.split(",") if t.strip()]
    return types


cache_dir = os.environ.get("NCOMPASS_CACHE_DIR")
profiler_type_str = os.environ.get("NCOMPASS_PROFILER_TYPE")
trace_name = os.environ.get("NCOMPASS_TRACE_NAME")

# Both must be set or both must be unset
if bool(cache_dir) != bool(profiler_type_str):
    missing = "NCOMPASS_PROFILER_TYPE" if cache_dir else "NCOMPASS_CACHE_DIR"
    present = "NCOMPASS_CACHE_DIR" if cache_dir else "NCOMPASS_PROFILER_TYPE"
    msg = f"Both NCOMPASS_CACHE_DIR and NCOMPASS_PROFILER_TYPE must be set. " \
          f"Found {present} but missing {missing}. Unset {present} to disable profiler."
    _log_error(cache_dir, msg)
    raise RuntimeError(msg)

if cache_dir and profiler_type_str:
    # Parse profiler types (handles aliases and comma-separated lists)
    profiler_types = _parse_profiler_types(profiler_type_str)

    if not profiler_types:
        msg = f"NCOMPASS_PROFILER_TYPE cannot be empty"
        _log_error(cache_dir, msg)
        raise ValueError(msg)

    # Validate each profiler type
    valid_options = ", ".join(VALID_PROFILER_TYPES)
    alias_options = ", ".join(PROFILER_ALIASES.keys())
    for pt in profiler_types:
        if pt not in VALID_PROFILER_TYPES:
            msg = f"Invalid profiler type '{pt}'. " \
                  f"Valid types: {valid_options}. " \
                  f"Valid aliases: {alias_options}. " \
                  f"Comma-separated lists are also supported (e.g., 'NVTX,CudaProfiler')."
            _log_error(cache_dir, msg)
            raise ValueError(msg)

    # Check if "NCU" is the profiler type - requires NCOMPASS_TRACE_NAME
    if "NCU" in profiler_types:
        if not trace_name:
            msg = "NCOMPASS_TRACE_NAME must be set when NCOMPASS_PROFILER_TYPE is 'NCU'. " \
                  "This specifies which trace file's kernel targets to use."
            _log_error(cache_dir, msg)
            raise RuntimeError(msg)

        # For NCU profiler type, the config is used by run_ncu_profile() to build
        # --kernel-id flags. The config is loaded via load_ncu_kernel_targets()
        # which reads NCOMPASS_CACHE_DIR and NCOMPASS_TRACE_NAME environment variables.
        #
        # Validate that the config file exists
        ncu_config_path = Path(cache_dir) / ".cache" / "ncompass" / "profiles" / ".default" / "NCU" / trace_name / "current" / "config.json"
        if not ncu_config_path.exists():
            msg = f"NCU config file not found: {ncu_config_path}. " \
                  f"Set kernel targets via the IDE or unset NCOMPASS_PROFILER_TYPE to disable."
            _log_error(cache_dir, msg)
            raise FileNotFoundError(msg)

        # NCU also enables NVTX rewrites for NVTX-based filtering in NCU
        # Add NVTX to the profiler types if not already present
        if "NVTX" not in profiler_types:
            profiler_types = profiler_types + ["NVTX"]

        # Remove NCU from the list for rewrite processing (NCU config is separate)
        profiler_types = [pt for pt in profiler_types if pt != "NCU"]
        # Fall through to process the remaining types (including NVTX)

    # Load configs from each profiler type directory (excluding NCU)
    rewrite_profiler_types = [pt for pt in profiler_types if pt != "NCU"]

    if rewrite_profiler_types:
        configs = []
        for pt in rewrite_profiler_types:
            config_path = Path(cache_dir) / f".cache/ncompass/profiles/.default/{pt}/current/config.json"

            if not config_path.exists():
                # Skip missing configs silently - allows partial configs
                # e.g., NSYS with only NVTX markers defined
                continue

            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                    configs.append(cfg)
            except Exception as e:
                _log_error(cache_dir, f"Failed to load config from {config_path}: {e}")
                raise

        if not configs:
            msg = f"No config files found for profiler types: {rewrite_profiler_types}. " \
                  f"Unset NCOMPASS_CACHE_DIR and NCOMPASS_PROFILER_TYPE to disable profiler."
            _log_error(cache_dir, msg)
            raise FileNotFoundError(msg)

        try:
            from ncompass.trace.core.rewrite import enable_rewrites
            from ncompass.trace.core.pydantic import RewriteConfig
            from ncompass.trace.core.config_manager import ConfigManager

            # Use ConfigManager to merge configs
            config_manager = ConfigManager(cache_dir)
            for cfg in configs:
                config_manager.add_config(cfg, merge=True)

            merged_cfg = config_manager.get_current_config()
            enable_rewrites(config=RewriteConfig.from_dict(merged_cfg))
        except Exception as e:
            _log_error(cache_dir, f"Failed to enable rewrites: {e}")
            raise
