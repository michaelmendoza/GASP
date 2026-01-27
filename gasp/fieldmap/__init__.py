"""
Field map compensation module for GASP.

This module provides tools for handling field map (B0) inhomogeneity effects
in GASP spectral filtering.
"""

from .multi_shifted import (
    FieldMapDictionary,
    train_fieldmap_dictionary,
    apply_gasp_with_fieldmap,
)

__all__ = [
    "FieldMapDictionary",
    "train_fieldmap_dictionary",
    "apply_gasp_with_fieldmap",
]
