"""Code slicing module for vulnerability detection"""

from .slicer import (
    SliceType,
    SliceConfig,
    CodeSlice,
    CodeSlicer,
    slice_batch,
)

__all__ = [
    'SliceType',
    'SliceConfig', 
    'CodeSlice',
    'CodeSlicer',
    'slice_batch',
]
