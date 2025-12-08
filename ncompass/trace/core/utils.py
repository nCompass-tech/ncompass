# Copyright 2025 nCompass Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Description: Utils for the core module."""

import importlib.util
from importlib.machinery import ModuleSpec
from typing import Optional, Union, Any, TYPE_CHECKING
from copy import deepcopy
import requests
import time
import sys
import gc
import os
from typing import Dict
from ncompass.trace.core.pydantic import ModuleConfig
from ncompass.trace.infra.utils import logger

if TYPE_CHECKING:
    from ncompass.trace.core.finder import RewritingFinder


def filepath_to_canonical_module_name(filepath: str) -> Optional[str]:
    """Convert a file path to its canonical module name using sys.path.
    
    This function finds the most specific sys.path entry that contains the file
    and derives the module name from the relative path. This ensures that modules
    are identified by their canonical import name rather than a path-derived name.
    
    For example:
        - If sys.path contains '/path/to/vllm_src'
        - And filepath is '/path/to/vllm_src/vllm/v1/worker/gpu_model_runner.py'
        - Returns 'vllm.v1.worker.gpu_model_runner'
    
    Args:
        filepath: Absolute or relative path to a Python file
        
    Returns:
        Canonical module name, or None if file is not under any sys.path entry
    """
    filepath = os.path.normpath(os.path.abspath(filepath))
    
    # Remove .py extension
    if filepath.endswith('.py'):
        module_path = filepath[:-3]
    else:
        module_path = filepath
    
    # Sort sys.path entries by length (longest first) to find most specific match
    # This ensures /path/to/vllm_src matches before /path/to
    sorted_paths = sorted(
        [os.path.normpath(os.path.abspath(p)) for p in sys.path if p],
        key=len,
        reverse=True
    )
    
    for path_entry in sorted_paths:
        if module_path.startswith(path_entry + os.sep):
            # Found a matching sys.path entry
            relative = module_path[len(path_entry) + 1:]  # +1 for separator
            # Convert path separators to dots
            module_name = relative.replace(os.sep, '.')
            # Handle __init__.py
            if module_name.endswith('.__init__'):
                module_name = module_name[:-9]
            return module_name
    
    return None


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

def markers_overlap(marker1: dict[str, Any], marker2: dict[str, Any]) -> bool:
    """Check if two markers overlap.
    
    Args:
        marker1: First marker with start_line and end_line
        marker2: Second marker with start_line and end_line
    
    Returns:
        True if markers overlap, False otherwise
    """
    start1: int = marker1['start_line']
    end1: int = marker1['end_line']
    start2: int = marker2['start_line']
    end2: int = marker2['end_line']
    
    # Check if marker2 starts within marker1 but extends beyond it
    if start2 >= start1 and start2 <= end1 and end2 > end1:
        return True
    
    # Check if marker1 starts within marker2 but extends beyond it
    if start1 >= start2 and start1 <= end2 and end1 > end2:
        return True
    
    return False


def merge_marker_configs(
    ai_configs: dict[str, Any],
    manual_configs: dict[str, Any]
) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Get the status of a request."""
    response = requests.get(f"{base_url}/status/{request_id}")
    data = response.json()
    return data

def submit_queue_request(
    request: dict[str, Any],
    base_url: str,
    endpoint: str,
    await_result: Optional[bool] = False
) -> Union[dict[str, Any], str]:
    """Submit a request to a queue.
    Args:
        request: Request to submit
        base_url: Base URL of the API
        endpoint: Endpoint to submit the request to
        await_result: Whether to await the result of the request
    
    Returns:
        Result of the request
    """
    resp = requests.post(f"{base_url}/{endpoint}", json=request)
    data = resp.json()
    request_id = data.get('request_id')
    if not request_id:
        raise ValueError(f"Failed to submit request to {base_url}/{endpoint}")
    if await_result:
        status = str(data.get('status'))
        response = data
        while status.lower() not in ['completed', 'failed']:
            response = get_request_status(request_id, base_url)
            status = str(response.get('status'))
            time.sleep(0.5)  # Prevent server overload
        if status.lower() == 'completed':
            return response['result']
        else:
            error = response.get('error') or f"Request failed: {status}"
            raise ValueError(f"Request failed: {error}")
    else:
        return request_id

def clear_cached_modules(targets: Dict[str, ModuleConfig]) -> Dict[str, Any]:
    """Clear cached modules and return old module references for updating.
    Returns:
        Dictionary mapping module names to their old module objects
    """
    old_modules = {}
    target_modules = list(targets.keys())
    if target_modules:
        for module_name in target_modules:
            old_module = None
            module_names_to_clear = []
            
            # First, try to find the module under the fully qualified name
            if module_name in sys.modules:
                old_module = sys.modules[module_name]
                module_names_to_clear.append(module_name)
                logger.debug(f"Clearing cached module: {module_name}")
            else:
                # Module not found under fully qualified name - might be imported locally
                # Check if it exists under the local name (last component of fully qualified name)
                local_name = module_name.split('.')[-1]
                if local_name in sys.modules:
                    candidate_module = sys.modules[local_name]
                    # For local imports, the module is stored under the local name
                    # We accept it as a match if it's a module object
                    if hasattr(candidate_module, '__name__') or hasattr(candidate_module, '__file__'):
                        old_module = candidate_module
                        module_names_to_clear.append(local_name)
                        logger.debug(f"Found module under local name '{local_name}', clearing it")
            
            if old_module is not None:
                # Store reference to old module using the fully qualified name as the key
                # This ensures update_module_references can find it
                old_modules[module_name] = old_module
                
                # Clear all entries pointing to this module
                for name in module_names_to_clear:
                    if name in sys.modules:
                        del sys.modules[name]
                
                # Also clear any submodules that might be cached
                modules_to_remove = [
                    name for name in list(sys.modules.keys())
                    if name == module_name or name.startswith(module_name + '.')
                ]
                for name in modules_to_remove:
                    if name in sys.modules:
                        del sys.modules[name]
            else:
                logger.debug(f"Module {module_name} not found in sys.modules (may not be imported yet)")
    
    return old_modules

def _should_skip_referrer(ref: Any, old_modules: Dict[str, Any], this_module: Any = None) -> bool:
    """Check if a referrer should be skipped when updating references.
    
    Args:
        ref: The referrer object to check
        old_modules: Dictionary of old modules (to skip)
        this_module: The current module (to skip its internals)
    
    Returns:
        True if the referrer should be skipped, False otherwise
    """
    if ref is old_modules or ref is sys.modules:
        return True
    if this_module is not None:
        if ref is this_module or getattr(ref, "__module__", None) == __name__:
            return True
    return False


def _update_dict_references(
    ref: dict,
    old_obj: Any,
    new_obj: Any,
    old_name: str,
    attr_name: Optional[str] = None
) -> None:
    """Update references in a dictionary namespace.
    
    Args:
        ref: Dictionary (typically a module's __dict__) to update
        old_obj: Old object to find and replace
        new_obj: New object to replace with
        old_name: Name of the module (for logging)
        attr_name: Optional attribute name (for logging)
    """
    for k, v in list(ref.items()):
        if v is old_obj:
            ref[k] = new_obj
            if attr_name:
                logger.debug(
                    f"Updated symbol ref {k} in {ref.get('__name__', 'dict')} "
                    f"from {old_name}.{attr_name} to new {old_name}.{attr_name}"
                )
            else:
                logger.debug(
                    f"Updated module ref {k} from old {old_name} to new {old_name}"
                )


def _update_direct_module_references(
    old_name: str,
    old_mod: Any,
    new_mod: Any,
    old_modules: Dict[str, Any]
) -> None:
    """Update direct module references (e.g., `import model`).
    
    Args:
        old_name: Name of the module
        old_mod: Old module object
        new_mod: New module object
        old_modules: Dictionary of old modules (for skipping)
    """
    for ref in gc.get_referrers(old_mod):
        if _should_skip_referrer(ref, old_modules):
            continue
        
        if isinstance(ref, dict):
            _update_dict_references(ref, old_mod, new_mod, old_name)


def _update_attribute_references(
    old_name: str,
    old_mod: Any,
    new_mod: Any,
    old_modules: Dict[str, Any],
    this_module: Any
) -> None:
    """Update from-imported symbol references (e.g., `from model import func`).
    
    Args:
        old_name: Name of the module
        old_mod: Old module object
        new_mod: New module object
        old_modules: Dictionary of old modules (for skipping)
        this_module: Current module (for skipping)
    """
    old_vars = getattr(old_mod, "__dict__", {})
    new_vars = getattr(new_mod, "__dict__", {})
    
    for attr_name, old_attr in list(old_vars.items()):
        # Only care about real attributes with a changed counterpart
        if attr_name.startswith("__"):
            continue
        if attr_name not in new_vars:
            continue
        
        new_attr = new_vars[attr_name]
        if new_attr is old_attr:
            continue  # not rewritten
        
        # Walk all referrers of the old attribute and swap to new_attr
        for ref in gc.get_referrers(old_attr):
            # Skip internals
            if ref is old_vars or ref is new_vars:
                continue
            if _should_skip_referrer(ref, old_modules, this_module):
                continue
            
            # Typical case: module/global/class namespace dict
            if isinstance(ref, dict):
                _update_dict_references(ref, old_attr, new_attr, old_name, attr_name)


def update_module_references(old_modules: Dict[str, Any]) -> None:
    """Update references to old modules and their attributes (incl. from x import y).

    This:
      1. Rebinds references to old module objects to the new modules.
      2. Rebinds references to old attributes (functions, classes, etc.)
         to the corresponding objects on the new module, so
         `from x import y` gets updated.
    
    Args:
        old_modules: Dictionary mapping module names to their old module objects
    """
    if not old_modules:
        return
    
    # Avoid patching our own internals
    this_module = sys.modules.get(__name__)
    
    for old_name, old_mod in old_modules.items():
        new_mod = sys.modules.get(old_name)
        if new_mod is None:
            continue
        
        # 1) Fix direct module references (e.g., `import model`)
        _update_direct_module_references(old_name, old_mod, new_mod, old_modules)
        
        # 2) Fix from-imported symbols (e.g., `from model import func`)
        _update_attribute_references(old_name, old_mod, new_mod, old_modules, this_module)


def _find_rewriting_finder() -> Optional['RewritingFinder']:  # type: ignore
    """Find the RewritingFinder from sys.meta_path.
    
    Returns:
        The RewritingFinder instance if found, None otherwise
    """
    from ncompass.trace.core.finder import RewritingFinder
    
    for finder in sys.meta_path:
        if isinstance(finder, RewritingFinder):
            return finder
    return None


def _resolve_module_file_path(
    module_name: str,
    module_config: ModuleConfig,
    old_modules: Dict[str, Any]
) -> Optional[str]:
    """Resolve the file path for a module.
    
    Args:
        module_name: Name of the module
        module_config: Configuration for the module
        old_modules: Dictionary of old module objects
    
    Returns:
        File path if found and exists, None otherwise
    """
    file_path = None
    
    # First check if old module has a __file__ attribute we can use
    if module_name in old_modules:
        old_mod = old_modules[module_name]
        if hasattr(old_mod, '__file__') and old_mod.__file__:
            file_path = old_mod.__file__
            logger.debug(f"Using file path from old module: {file_path}")
    
    # Fall back to config file path if available
    if not file_path or not os.path.exists(file_path):
        config_file_path = module_config.filePath
        if config_file_path and os.path.exists(config_file_path):
            file_path = config_file_path
            logger.debug(f"Using file path from config: {file_path}")
    
    if file_path and os.path.exists(file_path):
        return file_path
    return None


def _create_spec(
    module_name: str,
    file_path: str,
    rewriting_finder: Optional['RewritingFinder']  # type: ignore
) -> Optional[ModuleSpec]:
    """Create a module spec, optionally with rewriting enabled.
    
    Args:
        module_name: Name of the module
        file_path: Path to the module file
        rewriting_finder: Optional RewritingFinder instance
    
    Returns:
        ModuleSpec if created successfully, None otherwise
    """
    if rewriting_finder:
        # Use the RewritingFinder to create a rewriting spec
        from ncompass.trace.replacers.utils import create_replacer_from_config
        from ncompass.trace.core.loader import RewritingLoader
        
        # Get the merged config for this module
        merged_config = rewriting_finder.merged_configs.get(module_name)
        if merged_config:
            replacer = create_replacer_from_config(module_name, merged_config)
            spec = importlib.util.spec_from_loader(
                module_name,
                RewritingLoader(module_name, file_path, replacer),
                origin=file_path
            )
        else:
            # No config, use regular spec
            spec = importlib.util.spec_from_file_location(module_name, file_path)
    else:
        # No finder, use regular spec
        spec = importlib.util.spec_from_file_location(module_name, file_path)
    
    return spec


def _load_module_from_spec(module_name: str, spec: ModuleSpec) -> None:
    """Load a module from a spec.
    
    Args:
        module_name: Name of the module
        spec: ModuleSpec to load from
    """
    if spec and spec.loader:
        new_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = new_module
        spec.loader.exec_module(new_module)
        logger.debug(f"Re-imported module from file path with rewrites enabled: {module_name}")
    else:
        logger.warning(f"Failed to create spec for module {module_name}")


def _reimport_single_module(
    module_name: str,
    module_config: ModuleConfig,
    old_modules: Dict[str, Any],
    rewriting_finder: Optional['RewritingFinder']  # type: ignore
) -> None:
    """Reimport a single module.
    
    Args:
        module_name: Name of the module to reimport
        module_config: Configuration for the module
        old_modules: Dictionary of old module objects
        rewriting_finder: Optional RewritingFinder instance
    """
    # First, try to determine the canonical module name from the file path
    file_path = _resolve_module_file_path(module_name, module_config, old_modules)
    canonical_name = None
    if file_path:
        canonical_name = filepath_to_canonical_module_name(file_path)
        if canonical_name and canonical_name != module_name:
            logger.debug(f"Canonical module name for {module_name}: {canonical_name}")
    
    # Try importing with canonical name first (if different from module_name)
    import_succeeded = False
    if canonical_name and canonical_name != module_name:
        try:
            importlib.import_module(canonical_name)
            logger.debug(f"Re-imported module with canonical name: {canonical_name}")
            import_succeeded = True
        except Exception as e:
            logger.debug(f"Failed to import with canonical name {canonical_name}: {e}")
    
    # Try the original module name
    if not import_succeeded:
        try:
            importlib.import_module(module_name)
            logger.debug(f"Re-imported module with rewrites enabled: {module_name}")
            import_succeeded = True
        except Exception as e:
            logger.debug(f"Failed to import module {module_name}: {e}")
    
    # If both failed, try using the file path directly
    if not import_succeeded and file_path:
        try:
            # Use the canonical name if available, otherwise use the original name
            name_to_use = canonical_name if canonical_name else module_name
            spec = _create_spec(name_to_use, file_path, rewriting_finder)
            if spec:
                _load_module_from_spec(name_to_use, spec)
                logger.debug(f"Re-imported module from file path: {name_to_use}")
            else:
                logger.warning(f"Failed to create spec for module {name_to_use} from {file_path}")
        except Exception as e2:
            logger.warning(f"Failed to re-import module {module_name} from file path: {e2}")
    elif not import_succeeded:
        logger.warning(f"Failed to re-import module {module_name}: no valid file path available")


def reimport_modules(targets: Dict[str, ModuleConfig], old_modules: Dict[str, Any]) -> None:
    """Reimport modules and update references."""
    rewriting_finder = _find_rewriting_finder()
    
    for module_name, module_config in targets.items():
        _reimport_single_module(module_name, module_config, old_modules, rewriting_finder)
    
    # Update references in all loaded modules' namespaces
    update_module_references(old_modules)
