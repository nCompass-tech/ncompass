"""
ProfilingSession Iterative Workflow Example
"""

from dotenv import load_dotenv
load_dotenv()

from ncompass.trace.infra.utils import logger
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class ExampleConfig:
    """Example Configuration."""
    host_base: str = field(
        default_factory=lambda: f'{os.environ["HOME"]}/{os.environ.get("WORKDIR", "workspace")}'
    )
    torch_logs_dir: str = field(
        default_factory=lambda: f'{os.environ["HOME"]}/{os.environ.get("WORKDIR", "workspace")}/.cache/ncompass/torch_profile_logs'
    )
    profiling_session_dir: str = field(
        default_factory=lambda: f'{os.environ["HOME"]}/{os.environ.get("WORKDIR", "workspace")}/.cache/ncompass/sessions'
    )
    
    def __post_init__(self):
        """Create directories after initialization."""
        Path(self.torch_logs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.profiling_session_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Torch logs directory: {self.torch_logs_dir}")


# Default config instance
config = ExampleConfig()

