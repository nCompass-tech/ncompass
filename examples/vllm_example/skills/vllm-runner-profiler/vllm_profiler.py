#!/usr/bin/env python3
"""Start a vLLM server under nsys or ncu and run profiled workloads.

The server (and model) is loaded ONCE. Each scenario triggers a separate
cudaProfilerStart/Stop cycle, producing distinct capture ranges in the
same trace file. This avoids re-loading the model for every scenario.

Usage examples:
    # Single nsys trace with default settings:
    python vllm_profiler.py nsys --model Qwen/Qwen3.5-35B-A3B-FP8

    # Multiple scenarios (all in one trace, one model load):
    python vllm_profiler.py nsys --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --scenarios short_decode:10:256 long_prefill:2048:1 batch32:128:50:32

    # With custom vllm args:
    python vllm_profiler.py nsys --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --vllm-args "--moe-backend triton --attention-backend FLASHINFER" \
        --output flashinfer_trace

    # NCU roofline of a specific kernel:
    python vllm_profiler.py ncu --model Qwen/Qwen3.5-35B-A3B-FP8 \
        --kernel-name "device_kernel" --ncu-set roofline \
        --prompt-tokens 2048 --max-tokens 1
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerConfig:
    model: str
    host: str = "0.0.0.0"
    port: int = 8000
    vllm_args: str = ""
    max_wait_seconds: int = 900
    warmup_requests: int = 2


@dataclass(frozen=True)
class Scenario:
    """One profiling scenario → one capture range in the trace."""
    name: str
    prompt_tokens: int = 10
    max_tokens: int = 50
    batch_size: int = 1


@dataclass(frozen=True)
class NsysConfig:
    extra_nsys_args: str = ""


@dataclass(frozen=True)
class NcuConfig:
    kernel_name: str = ""
    ncu_set: str = "roofline"
    launch_skip: int = 3
    launch_count: int = 1


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _base_url(cfg: ServerConfig) -> str:
    return f"http://localhost:{cfg.port}"


def _health_check(cfg: ServerConfig) -> bool:
    try:
        urlopen(f"{_base_url(cfg)}/health", timeout=5)
        return True
    except (URLError, OSError):
        return False


def _wait_ready(cfg: ServerConfig) -> None:
    print(f"  Waiting for server (max {cfg.max_wait_seconds}s)...", end="", flush=True)
    waited = 0
    while waited < cfg.max_wait_seconds:
        if _health_check(cfg):
            print(f" ready ({waited}s)")
            return
        time.sleep(5)
        waited += 5
        print(".", end="", flush=True)
    raise TimeoutError(f"Server not ready after {cfg.max_wait_seconds}s")


def _send_request(cfg: ServerConfig, prompt: str, max_tokens: int) -> None:
    payload = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    req = Request(
        f"{_base_url(cfg)}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    urlopen(req, timeout=300)


def _post(cfg: ServerConfig, path: str) -> None:
    req = Request(f"{_base_url(cfg)}{path}", method="POST")
    urlopen(req, timeout=30)


def _warmup(cfg: ServerConfig) -> None:
    print(f"  Warming up ({cfg.warmup_requests} requests)...")
    for _ in range(cfg.warmup_requests):
        _send_request(cfg, "Hello", 10)


def _make_prompt(token_count: int) -> str:
    return "word " * token_count


def _kill_server(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        pass


def _kill_leftover_servers() -> None:
    subprocess.run(["pkill", "-f", "vllm serve"], capture_output=True)
    time.sleep(2)


# ---------------------------------------------------------------------------
# Run a single scenario against a running server
# ---------------------------------------------------------------------------

def _run_scenario(cfg: ServerConfig, scenario: Scenario) -> None:
    """Run one profiled scenario: start_profile → workload → stop_profile."""
    print(f"\n  --- {scenario.name} ---")
    print(f"  prompt_tokens={scenario.prompt_tokens}, max_tokens={scenario.max_tokens}, "
          f"batch_size={scenario.batch_size}")

    _post(cfg, "/start_profile")

    prompt = _make_prompt(scenario.prompt_tokens)
    start = time.perf_counter()
    for _ in range(scenario.batch_size):
        _send_request(cfg, prompt, scenario.max_tokens)
    elapsed = time.perf_counter() - start

    _post(cfg, "/stop_profile")
    print(f"  Done: {elapsed*1000:.1f}ms")


# ---------------------------------------------------------------------------
# nsys profiling
# ---------------------------------------------------------------------------

def run_nsys(
    cfg: ServerConfig,
    scenarios: list[Scenario],
    trace_dir: Path,
    output_name: str,
    nsys_cfg: NsysConfig,
) -> Path:
    """Launch server once under nsys, run all scenarios, shut down."""
    trace_path = str(trace_dir / output_name)

    cmd = [
        "ncompass", "profile", "--nsys",
        "-o", trace_path,
    ]
    if nsys_cfg.extra_nsys_args:
        cmd.extend(nsys_cfg.extra_nsys_args.split())
    cmd.append("--")
    cmd.extend(["vllm", "serve", cfg.model, "--host", cfg.host, "--port", str(cfg.port)])
    if cfg.vllm_args:
        cmd.extend(cfg.vllm_args.split())
    cmd.extend(["--profiler-config.profiler", "cuda"])

    print(f"\n{'='*60}")
    print(f"  nsys profiling: {len(scenarios)} scenario(s)")
    print(f"  Output: {trace_path}.nsys-rep")
    print(f"{'='*60}")

    _kill_leftover_servers()
    proc = subprocess.Popen(cmd)

    try:
        _wait_ready(cfg)
        _warmup(cfg)

        for scenario in scenarios:
            _run_scenario(cfg, scenario)
            time.sleep(2)  # brief pause between captures

    finally:
        print("\n  Shutting down server...")
        time.sleep(5)
        _kill_server(proc.pid)

    rep = Path(f"{trace_path}.nsys-rep")
    if rep.exists():
        print(f"\n  Trace: {rep} ({rep.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"\n  WARNING: {rep} not found")
    return rep


# ---------------------------------------------------------------------------
# ncu profiling
# ---------------------------------------------------------------------------

def run_ncu(
    cfg: ServerConfig,
    scenario: Scenario,
    trace_dir: Path,
    output_name: str,
    ncu_cfg: NcuConfig,
) -> Path:
    """Launch server under ncu, run one scenario, shut down."""
    output_path = str(trace_dir / output_name)

    cmd = [
        "ncu",
        "--profile-from-start", "off",
        "--target-processes", "all",
        "--set", ncu_cfg.ncu_set,
    ]
    if ncu_cfg.kernel_name:
        cmd.extend(["--kernel-name", ncu_cfg.kernel_name])
    cmd.extend(["--launch-skip", str(ncu_cfg.launch_skip)])
    cmd.extend(["--launch-count", str(ncu_cfg.launch_count)])
    cmd.extend(["-o", output_path, "-f"])
    cmd.extend(["vllm", "serve", cfg.model, "--host", cfg.host, "--port", str(cfg.port)])
    if cfg.vllm_args:
        cmd.extend(cfg.vllm_args.split())
    cmd.extend(["--profiler-config.profiler", "cuda"])

    print(f"\n{'='*60}")
    print(f"  ncu profiling: {scenario.name}")
    print(f"  kernel={ncu_cfg.kernel_name or '(all)'}, set={ncu_cfg.ncu_set}")
    print(f"{'='*60}")

    _kill_leftover_servers()
    proc = subprocess.Popen(cmd)

    try:
        _wait_ready(cfg)
        _warmup(cfg)
        _run_scenario(cfg, scenario)
    finally:
        print("\n  Shutting down server...")
        time.sleep(5)
        _kill_server(proc.pid)

    rep = Path(f"{output_path}.ncu-rep")
    if rep.exists():
        print(f"\n  Report: {rep} ({rep.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"\n  WARNING: {rep} not found")
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_scenario(spec: str) -> Scenario:
    """Parse 'name:prompt_tokens:max_tokens[:batch_size]'."""
    parts = spec.split(":")
    if len(parts) < 3:
        raise ValueError(f"Scenario format: name:prompt_tokens:max_tokens[:batch_size], got: {spec}")
    return Scenario(
        name=parts[0],
        prompt_tokens=int(parts[1]),
        max_tokens=int(parts[2]),
        batch_size=int(parts[3]) if len(parts) > 3 else 1,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Profile vLLM server with nsys or ncu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="profiler", required=True)

    # --- shared args for both subcommands ---
    for name, help_text in [("nsys", "Profile with nsys"), ("ncu", "Profile with ncu")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--model", required=True, help="HuggingFace model name")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--trace-dir", type=Path, default=Path("traces"))
        p.add_argument("--output", type=str, default="trace", help="Output trace name")
        p.add_argument("--vllm-args", type=str, default="", help="Extra args for 'vllm serve'")
        p.add_argument("--warmup", type=int, default=2, help="Warmup requests")
        p.add_argument("--max-wait", type=int, default=900, help="Max seconds to wait for server")

        # single scenario shorthand
        p.add_argument("--prompt-tokens", type=int, default=10)
        p.add_argument("--max-tokens", type=int, default=50)
        p.add_argument("--batch-size", type=int, default=1)

        # multi-scenario
        p.add_argument("--scenarios", nargs="+", metavar="name:ptok:mtok[:batch]",
                        help="Multiple scenarios (overrides single-scenario args)")

    # --- ncu-specific ---
    ncu_p = sub.choices["ncu"]
    ncu_p.add_argument("--kernel-name", type=str, default="")
    ncu_p.add_argument("--ncu-set", type=str, default="roofline")
    ncu_p.add_argument("--launch-skip", type=int, default=3)
    ncu_p.add_argument("--launch-count", type=int, default=1)

    # --- nsys-specific ---
    nsys_p = sub.choices["nsys"]
    nsys_p.add_argument("--nsys-args", type=str, default="", help="Extra args for nsys")

    args = parser.parse_args()
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    server_cfg = ServerConfig(
        model=args.model,
        port=args.port,
        vllm_args=args.vllm_args,
        max_wait_seconds=args.max_wait,
        warmup_requests=args.warmup,
    )

    # Build scenario list
    if args.scenarios:
        scenarios = [_parse_scenario(s) for s in args.scenarios]
    else:
        scenarios = [Scenario(
            name=args.output,
            prompt_tokens=args.prompt_tokens,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
        )]

    if args.profiler == "nsys":
        run_nsys(server_cfg, scenarios, args.trace_dir, args.output, NsysConfig(
            extra_nsys_args=args.nsys_args,
        ))
    else:
        # NCU: one scenario at a time (ncu captures specific kernel launches)
        for scenario in scenarios:
            run_ncu(server_cfg, scenario, args.trace_dir, args.output, NcuConfig(
                kernel_name=args.kernel_name,
                ncu_set=args.ncu_set,
                launch_skip=args.launch_skip,
                launch_count=args.launch_count,
            ))

    print(f"\nAll done. Traces in: {args.trace_dir}")


if __name__ == "__main__":
    main()
