"""
ProfilingSession Iterative Workflow Example
"""

from dotenv import load_dotenv
load_dotenv()

from ncompass.trace.infra.utils import logger
import logging
import os
from pathlib import Path

logger.setLevel(logging.DEBUG)


HOST_BASE = f'{os.environ["HOME"]}/{os.environ.get("WORKDIR", "workspace")}'
TORCH_LOGS_DIR = f"{HOST_BASE}/.cache/ncompass/torch_profile_logs"
PROFILING_SESSION_DIR = f"{HOST_BASE}/.cache/ncompass/sessions"
Path(TORCH_LOGS_DIR).mkdir(parents=True, exist_ok=True)
Path(PROFILING_SESSION_DIR).mkdir(parents=True, exist_ok=True)
logger.info(f"Torch logs directory: {TORCH_LOGS_DIR}")
