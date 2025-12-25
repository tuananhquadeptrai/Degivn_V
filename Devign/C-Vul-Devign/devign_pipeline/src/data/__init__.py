"""Data loading and exploration utilities for Devign dataset"""

# Lazy imports - modules are imported only when accessed
__all__ = [
    "DevignLoader",
    "DevignSample", 
    "DataLoader",
    "load_jsonl",
    "load_dataset",
]

def __getattr__(name):
    if name in __all__:
        from . import loader
        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
