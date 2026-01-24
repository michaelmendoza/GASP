"""
GASP ML Module: Neural network approaches for multi-tissue GASP.

This module provides machine learning extensions to GASP for handling
heterogeneous multi-tissue data with varying T2/T1 ratios.
"""

from .conditional_gasp import ConditionalGASP
from .data_generator import generate_training_batch, generate_training_dataset
from .trainer import train_conditional_gasp
from .losses import conditional_gasp_loss, spectral_profile_loss

__all__ = [
    'ConditionalGASP',
    'generate_training_batch',
    'generate_training_dataset',
    'train_conditional_gasp',
    'conditional_gasp_loss',
    'spectral_profile_loss',
]
