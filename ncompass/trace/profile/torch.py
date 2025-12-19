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

"""
Description: Torch profiler context managers for AST rewriting.
"""

from typing import Optional, Any, List, Tuple, Callable
import torch
import inspect
from ncompass.trace.profile.base import ProfileContextBase
from ncompass.types.immutable import mutate
from torch.utils.flop_counter import FlopCounterMode

class TorchRecordContext(ProfileContextBase):
    """Context manager for Torch profiler record_function."""
    def __init__(self, name: str) -> None:
        """Initialize Torch profiler context with a name."""
        self.name = name
        self.context = torch.profiler.record_function(self.name)

    def __enter__(self) -> Any:
        """Enter the profiler context."""
        return self.context.__enter__()
    
    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """Exit the profiler context."""
        return self.context.__exit__(exc_type, exc_value, traceback)

class TorchFlopCounterContext(ProfileContextBase):
    """Context manager for Torch flop counter with parametric tensor size sweeping.
    
    This context manager runs provided code multiple times with different tensor sizes,
    collecting FLOP counts for each size. Useful for understanding how FLOPs scale with
    input dimensions.
    
    Args:
        name: Label for this profiling operation
        var_name: Name of the tensor variable to scale (as it appears in calling scope)
        size_multipliers: List of scaling factors to apply to tensor dimensions.
                         Defaults to [0.1, 0.3, 0.5, 0.9, 1, 1.1, 2, 3, 10]
        scale_dims: Tuple of dimension indices to scale. None means scale all dimensions.
                   Example: (0,) scales only batch dimension, (1, 2) scales spatial dims
    
    Usage Pattern 1 - With callable:
        X = torch.randn(128, 784)
        with TorchFlopCounterContext(name="inference", var_name="X") as ctx:
            ctx.run(lambda: model(X))
    
    Usage Pattern 2 - With iteration:
        X = torch.randn(128, 784)
        with TorchFlopCounterContext(name="inference", var_name="X") as ctx:
            for mult, X_scaled in ctx:
                output = model(X_scaled)
                ctx.record_flops()
    """
    
    DEFAULT_MULTIPLIERS = [0.1, 0.3, 0.5, 0.9, 1, 1.1, 2, 3, 10]
    
    def __init__(
        self,
        name: str,
        var_name: str,
        size_multipliers: Optional[List[float]] = None,
        scale_dims: Optional[Tuple[int, ...]] = None
    ) -> None:
        """Initialize Torch flop counter context with parametric sizing."""
        self.name = name
        self.var_name = var_name
        self.size_multipliers = size_multipliers if size_multipliers is not None else self.DEFAULT_MULTIPLIERS
        self.scale_dims = scale_dims
        
        # Storage for frame inspection and state
        self.calling_frame = None
        self.original_tensor = None
        self.original_tensor_copy = None
        self.flop_results = []  # Store (multiplier, shape, flop_count) tuples
        self.current_multiplier_idx = 0
        self.current_flop_mode = None
        self.current_multiplier = None
        self.current_shape = None

    @mutate
    def __enter__(self) -> 'TorchFlopCounterContext':
        """Enter the profiler context and setup for parametric execution."""
        # Get the calling frame to access local variables
        # Note: @mutate decorator adds an extra frame, so we need to go back 2 levels
        frame = inspect.currentframe()
        if frame is not None:
            # Go back through: current frame -> mutate wrapper -> actual caller
            self.calling_frame = frame.f_back.f_back if frame.f_back else None
        else:
            self.calling_frame = None
        
        if self.calling_frame is None:
            raise RuntimeError("Could not access calling frame")
        
        # Look up the tensor variable in the calling scope
        if self.var_name not in self.calling_frame.f_locals:
            raise ValueError(f"Variable '{self.var_name}' not found in calling scope")
        
        self.original_tensor = self.calling_frame.f_locals[self.var_name]
        
        if not isinstance(self.original_tensor, torch.Tensor):
            raise TypeError(f"Variable '{self.var_name}' must be a torch.Tensor, got {type(self.original_tensor)}")
        
        # Make a copy to restore later
        self.original_tensor_copy = self.original_tensor.clone()
        
        print(f"\n[NC] FLOP counting for '{self.name}' with variable '{self.var_name}'")
        print(f"[NC] Original tensor shape: {list(self.original_tensor.shape)}")
        print(f"[NC] Testing {len(self.size_multipliers)} size configurations...")
        print(f"[NC] Multipliers: {self.size_multipliers}")
        if self.scale_dims is not None:
            print(f"[NC] Scaling dimensions: {self.scale_dims}")
        else:
            print(f"[NC] Scaling all dimensions")
        print("")
        
        return self
    
    @mutate
    def __iter__(self):
        """Make the context manager iterable over scaled tensors."""
        self.current_multiplier_idx = 0
        return self
    
    @mutate
    def __next__(self) -> Tuple[float, torch.Tensor]:
        """Yield next scaled tensor with its multiplier.
        
        Returns:
            Tuple of (multiplier, scaled_tensor)
        """
        # Close previous FLOP mode if active
        if self.current_flop_mode is not None:
            self.record_flops()
        
        if self.current_multiplier_idx >= len(self.size_multipliers):
            raise StopIteration
        
        multiplier = self.size_multipliers[self.current_multiplier_idx]
        self.current_multiplier_idx += 1
        
        # Create scaled tensor
        scaled_tensor = self._create_scaled_tensor(multiplier)
        
        print(f"[NC] Testing multiplier {multiplier:.2f}, shape {list(scaled_tensor.shape)}")
        
        # Start FLOP counting for this iteration
        self.current_flop_mode = FlopCounterMode(display=False)
        self.current_flop_mode.__enter__()
        self.current_multiplier = multiplier
        self.current_shape = list(scaled_tensor.shape)
        
        return multiplier, scaled_tensor
    
    @mutate
    def record_flops(self) -> None:
        """Record FLOP count for the current iteration."""
        if self.current_flop_mode is not None:
            self.current_flop_mode.__exit__(None, None, None)
            total_flops = self.current_flop_mode.get_total_flops()
            self.flop_results.append((self.current_multiplier, self.current_shape, total_flops))
            print(f"[NC]   Total FLOPs: {total_flops:,}")
            self.current_flop_mode = None
    
    @mutate
    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> None:
        """Exit the profiler context and print summary."""
        # Restore original tensor
        if self.calling_frame is not None and self.original_tensor_copy is not None:
            # Note: Modifying f_locals doesn't always work for restoration in all Python versions
            # The original tensor object still exists, so we update it in-place
            self.original_tensor.data = self.original_tensor_copy.data
            self.original_tensor.requires_grad = self.original_tensor_copy.requires_grad
        
        # Print summary table
        print(f"\n[NC] FLOP counting summary for '{self.name}':")
        print(f"[NC] {'Multiplier':<12} {'Shape':<30} {'FLOPs':<20}")
        print(f"[NC] {'-'*12} {'-'*30} {'-'*20}")
        
        for multiplier, shape, flops in self.flop_results:
            shape_str = str(list(shape))
            flops_str = f"{flops:,}" if isinstance(flops, int) else str(flops)
            print(f"[NC] {multiplier:<12.2f} {shape_str:<30} {flops_str:<20}")
        
        print("")
        
        return None
    
    def _create_scaled_tensor(self, multiplier: float) -> torch.Tensor:
        """Create a scaled version of the original tensor with random data.
        
        Args:
            multiplier: Scaling factor for dimensions
            
        Returns:
            New tensor with scaled dimensions, filled with random data in the same range
        """
        # Always scale from the original tensor shape, not the current modified one
        original_shape = list(self.original_tensor_copy.shape)
        new_shape = original_shape.copy()
        
        # Determine which dimensions to scale
        dims_to_scale = self.scale_dims if self.scale_dims is not None else range(len(new_shape))
        
        # Apply scaling to specified dimensions
        for dim in dims_to_scale:
            if dim >= len(new_shape):
                raise ValueError(f"Dimension {dim} out of range for tensor with {len(new_shape)} dimensions")
            new_shape[dim] = max(1, int(new_shape[dim] * multiplier))
        
        # Get min/max from original tensor copy for random sampling
        with torch.no_grad():
            min_val = self.original_tensor_copy.min().item()
            max_val = self.original_tensor_copy.max().item()
        
        # Create new tensor with scaled shape
        scaled_tensor = torch.empty(
            new_shape,
            dtype=self.original_tensor.dtype,
            device=self.original_tensor.device,
            requires_grad=self.original_tensor.requires_grad
        )
        
        # Fill with random data in the same range as original
        with torch.no_grad():
            scaled_tensor.uniform_(min_val, max_val)
        
        return scaled_tensor
    
    def _inject_tensor(self, tensor: torch.Tensor) -> None:
        """Inject a tensor into the calling frame's local scope.
        
        Note: Direct f_locals modification doesn't always work in Python,
        so we modify the original tensor object in-place instead.
        """
        # Update the original tensor object in-place
        # This ensures the calling code sees the updated tensor
        self.original_tensor.data = tensor.data
        self.original_tensor.requires_grad = tensor.requires_grad
    
    @mutate
    def run(self, func: Callable[[], Any]) -> List[Tuple[float, List[int], int]]:
        """Run a callable function with multiple tensor sizes and collect FLOP counts.
        
        This is the recommended way to use TorchFlopCounterContext for parametric profiling.
        
        Args:
            func: Callable that uses the tensor variable from the calling scope.
                  The tensor will be automatically scaled and injected before each call.
            
        Returns:
            List of (multiplier, shape, flop_count) tuples
            
        Example:
            X = torch.randn(128, 784)
            with TorchFlopCounterContext(name="inference", var_name="X") as ctx:
                ctx.run(lambda: model(X))
        """
        self.flop_results = []
        
        for multiplier in self.size_multipliers:
            # Create scaled tensor
            scaled_tensor = self._create_scaled_tensor(multiplier)
            
            # Inject it into the calling scope
            self._inject_tensor(scaled_tensor)
            
            print(f"[NC] Running with multiplier {multiplier:.2f}, shape {list(scaled_tensor.shape)}")
            
            # Run with FLOP counter
            flop_mode = FlopCounterMode(display=False)
            with flop_mode:
                func()
            
            # Collect FLOP count
            total_flops = flop_mode.get_total_flops()
            self.flop_results.append((multiplier, list(scaled_tensor.shape), total_flops))
            
            print(f"[NC]   Total FLOPs: {total_flops:,}")
        
        return self.flop_results