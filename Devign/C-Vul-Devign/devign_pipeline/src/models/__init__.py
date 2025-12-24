"""Models module for vulnerability detection."""

from .slice_attention_bigru import SliceAttBiGRU, create_model

__all__ = ["SliceAttBiGRU", "create_model"]
