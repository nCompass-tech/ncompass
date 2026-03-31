"""GPU exclusive lock for benchmark/profiling isolation.

Uses fcntl.flock() on a shared lock file mounted from the host. The lock
is automatically released when the file descriptor is closed (including
process crash/kill), so there is no stale-lock problem.

Lock file path: /gpu_locks/gpu0.lock (container) or GPU_LOCK_PATH env override.
Timeout: 600s default or GPU_LOCK_TIMEOUT env override.
"""

from __future__ import annotations

import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_LOCK_PATH = "/gpu_locks/gpu0.lock"
DEFAULT_TIMEOUT = 600  # 10 minutes


@contextmanager
def gpu_lock(purpose: str = ""):
    """Hold an exclusive GPU lock. Blocks until acquired or timeout.

    Gracefully degrades: if the lock directory does not exist (e.g. running
    locally without Docker, or single-agent), the lock is silently skipped.
    """
    # If a parent process already holds the lock (e.g. gpu-lock shell wrapper
    # ran bench.py), skip to avoid deadlock.
    if os.environ.get("GPU_LOCK_HELD") == "1":
        yield
        return

    lock_file = Path(os.environ.get("GPU_LOCK_PATH", DEFAULT_LOCK_PATH))
    timeout = float(os.environ.get("GPU_LOCK_TIMEOUT", DEFAULT_TIMEOUT))

    # No lock directory → not configured, proceed without lock.
    if not lock_file.parent.exists():
        yield
        return

    label = f" ({purpose})" if purpose else ""
    container = os.environ.get("HOSTNAME", "unknown")[:12]

    fd = open(lock_file, "w")

    print(f"[gpu-lock] Acquiring GPU lock{label} ...")

    start = time.monotonic()
    last_log = start

    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (BlockingIOError, OSError):
            elapsed = time.monotonic() - start
            if elapsed >= timeout:
                fd.close()
                print(
                    f"[gpu-lock] TIMEOUT after {elapsed:.0f}s waiting for GPU lock{label}.",
                    file=sys.stderr,
                )
                sys.exit(124)
            if time.monotonic() - last_log >= 30:
                print(f"[gpu-lock] Waiting for GPU lock{label}... ({elapsed:.0f}s)")
                last_log = time.monotonic()
            time.sleep(1.0)

    wait_time = time.monotonic() - start
    # Write holder info for debugging (`cat /gpu_locks/gpu0.lock`).
    fd.seek(0)
    fd.truncate()
    fd.write(f"holder={container} purpose={purpose} since={time.time():.0f}\n")
    fd.flush()

    if wait_time > 1.0:
        print(f"[gpu-lock] Acquired GPU lock{label} after {wait_time:.1f}s wait")
    else:
        print(f"[gpu-lock] Acquired GPU lock{label}")

    try:
        os.environ["GPU_LOCK_HELD"] = "1"
        yield
    finally:
        os.environ.pop("GPU_LOCK_HELD", None)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        print(f"[gpu-lock] Released GPU lock{label}")
