"""
Description: NVTX utils for AST rewriting.
"""


from ncompasslib.trait import Trait


class _ProfileContextBase(Trait):
    """Context manager for profiling."""
    def __init__(self):
        raise NotImplementedError("Subclasses must implement __init__")

    def __enter__(self):
        raise NotImplementedError("Subclasses must implement __enter__")
    
    def __exit__(self, exc_type, exc_value, traceback):
        raise NotImplementedError("Subclasses must implement __exit__")
