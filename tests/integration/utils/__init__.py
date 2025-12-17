"""Utility modules for integration tests."""

from tests.integration.utils.docker import (
    build_docker_image,
    run_docker_command,
    create_symlink,
    remove_symlink,
)

from tests.integration.utils.compare import (
    compare_json_files,
    compare_files_binary,
    load_json_file,
    normalize_json_data,
    sort_trace_events,
)

from tests.integration.utils.trace_validation import (
    count_events_by_category_json,
)

__all__ = [
    # Docker utilities
    "build_docker_image",
    "run_docker_command",
    "create_symlink",
    "remove_symlink",
    # File comparison utilities
    "compare_json_files",
    "compare_files_binary",
    "load_json_file",
    "normalize_json_data",
    "sort_trace_events",
    # Trace validation utilities
    "count_events_by_category_json",
]
