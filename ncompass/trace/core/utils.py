"""Description: Utils for the core module."""

import importlib.util
from typing import Optional, Union
from copy import deepcopy
from ncompass.trace.infra.utils import logger
import requests
import time


def extract_source_code(target_module: str) -> Optional[str]:
    """Extract source code from a module."""
    try:
        spec = importlib.util.find_spec(target_module)
        if not spec or not spec.origin:
            logger.error(f"Could not find module {target_module}")
            return None
        
        file_path = spec.origin
        with open(file_path, 'r') as f:
            source_code = f.read()
        return source_code
    except Exception as e:
        logger.error(f"Error loading module {target_module}: {e}")
        return None

def extract_code_region(target_module: str, start_line: int, end_line: int) -> Optional[str]:
    """Extract a region of code from a module."""
    source_code = extract_source_code(target_module)
    if source_code is None:
        return None
    else:
        lines = source_code.split('\n')
        # Line nos are 1-indexed
        code_region = '\n'.join(lines[max(0, start_line-1):min(len(lines), end_line)])
        return code_region

def markers_overlap(marker1: dict[str, object], marker2: dict[str, object]) -> bool:
    """Check if two markers overlap.
    
    Args:
        marker1: First marker with start_line and end_line
        marker2: Second marker with start_line and end_line
    
    Returns:
        True if markers overlap, False otherwise
    """
    start1, end1 = marker1['start_line'], marker1['end_line']
    start2, end2 = marker2['start_line'], marker2['end_line']
    
    # Check if marker2 starts within marker1 but extends beyond it
    if start2 >= start1 and start2 <= end1 and end2 > end1:
        return True
    
    # Check if marker1 starts within marker2 but extends beyond it
    if start1 >= start2 and start1 <= end2 and end1 > end2:
        return True
    
    return False


def merge_marker_configs(
    ai_configs: dict[str, object],
    manual_configs: dict[str, object]
) -> dict[str, object]:
    """Merge AI and manual configs, with manual taking priority on conflicts.
    
    For each file, if there are conflicting markers (overlapping but not subset),
    the manual marker takes priority and the conflicting AI marker is discarded.
    
    Args:
        ai_configs: AI-generated configurations
        manual_configs: Manual configurations
    
    Returns:
        Merged configuration with conflicts resolved
    """
    merged = deepcopy(manual_configs)
    
    for filepath, ai_config in ai_configs.items():
        if filepath not in merged:
            # No manual config for this file, use AI config as-is
            merged[filepath] = ai_config
            continue
        
        # File has both AI and manual configs - need to check for conflicts
        manual_wrappings = merged[filepath].get('func_line_range_wrappings', [])
        ai_wrappings = ai_config.get('func_line_range_wrappings', [])
        
        # Filter out AI markers that conflict with manual markers
        filtered_ai_wrappings = []
        for ai_marker in ai_wrappings:
            conflicts = False
            for manual_marker in manual_wrappings:
                if markers_overlap(ai_marker, manual_marker):
                    logger.debug(
                        f"[Finder] Discarding conflicting AI marker in {filepath}: "
                        f"{ai_marker['function']} ({ai_marker['start_line']}-{ai_marker['end_line']}) "
                        f"conflicts with manual marker ({manual_marker['start_line']}-{manual_marker['end_line']})"
                    )
                    conflicts = True
                    break
            
            if not conflicts:
                filtered_ai_wrappings.append(ai_marker)
        
        # Add non-conflicting AI markers to manual markers
        if filtered_ai_wrappings:
            merged[filepath]['func_line_range_wrappings'] = manual_wrappings + filtered_ai_wrappings
            logger.debug(
                f"Merged {len(filtered_ai_wrappings)} AI markers with "
                f"{len(manual_wrappings)} manual markers for {filepath}"
            )
    
    return merged

def get_request_status(
    request_id: str,
    base_url: str
) -> dict[str, object]:
    """Get the status of a request."""
    response = requests.get(f"{base_url}/status/{request_id}")
    data = response.json()
    return data

def submit_queue_request(
    request: dict[str, object],
    base_url: str,
    endpoint: str,
    await_result: Optional[bool] = False
) -> Union[dict[str, object], str]:
    """Submit a request to a queue.
    Args:
        request: Request to submit
        base_url: Base URL of the API
        endpoint: Endpoint to submit the request to
        await_result: Whether to await the result of the request
    
    Returns:
        Result of the request
    """
    response = requests.post(f"{base_url}/{endpoint}", json=request)
    data = response.json()
    request_id = data.get('request_id')
    if not request_id:
        raise ValueError(f"Failed to submit request to {base_url}/{endpoint}")
    if await_result:
        status = str(data.get('status'))
        while status.lower() not in ['completed', 'failed']:
            response = get_request_status(request_id, base_url)
            status = str(response.get('status'))
            time.sleep(0.5)  # Prevent server overload
        if status.lower() == 'completed':
            return response['result']
        else:
            error = response.get('error')
            raise ValueError(f"Request failed: {error}")
    else:
        return request_id