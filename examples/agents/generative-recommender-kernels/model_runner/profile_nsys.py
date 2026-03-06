#!/usr/bin/env python3
"""Run run_model.py under nsys profiling."""
import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NsysConfig:
    script: Path
    traces_dir: Path
    default_output: str = "report1"

    @property
    def default_output_path(self) -> str:
        return str(self.traces_dir / self.default_output)


def parse_args(config: NsysConfig) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run run_model.py under nsys profiling. "
                    "All unrecognized args are forwarded to run_model.py.",
    )
    parser.add_argument("-o", "--output", type=str, default=config.default_output_path,
                        help=f"nsys output path (default: {config.traces_dir}/report1)")
    args, remaining = parser.parse_known_args()
    return args.output, remaining


def build_nsys_cmd(config: NsysConfig, output: str, model_args: list[str]) -> list[str]:
    return [
        "nsys", "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--cuda-memory-usage=true",
        "-o", output,
        "--force-overwrite=true",
        sys.executable, "-u", str(config.script),
    ] + model_args


def main():
    script_dir = Path(__file__).resolve().parent
    config = NsysConfig(
        script=script_dir / "run_model.py",
        traces_dir=script_dir / ".nsys_traces",
    )

    output, model_args = parse_args(config)
    config.traces_dir.mkdir(exist_ok=True)

    cmd = build_nsys_cmd(config, output, model_args)
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
