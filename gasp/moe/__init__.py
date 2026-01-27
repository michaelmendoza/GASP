"""Mixture of Experts GASP (MoE-GASP).

This submodule implements MoE-GASP, an extension of GASP that uses multiple
specialized expert models for different T2/T1 regimes. A gating network
learns to route input signals to the appropriate experts.

Key Components
--------------
MoEGASPConfig : Configuration dataclass for MoE-GASP
MoEGASPModel : Container for trained model components
train_moe_gasp : Train MoE-GASP on synthetic data
run_moe_gasp : Apply trained model to new data

Example
-------
>>> import numpy as np
>>> from gasp.moe import (
...     train_moe_gasp, run_moe_gasp, MoEGASPConfig,
...     generate_ssfp_training_data
... )
>>>
>>> # Generate training data
>>> signals, t2_t1, params = generate_ssfp_training_data(
...     n_samples=5000, npcs=16, seed=42
... )
>>>
>>> # Define desired profile (e.g., Gaussian)
>>> desired = np.exp(-np.linspace(-2, 2, 16)**2)
>>>
>>> # Configure and train
>>> config = MoEGASPConfig(n_experts=5, gating_type='mlp')
>>> model = train_moe_gasp(signals, desired, t2_t1, config)
>>>
>>> # Apply to new data
>>> output, weights = run_moe_gasp(new_ssfp_data, model)
"""

from gasp.moe.moe_gasp import (
    MoEGASPConfig,
    MoEGASPModel,
    train_moe_gasp,
    run_moe_gasp,
    train_moe_gasp_from_image,
    train_moe_gasp_with_coils,
    analyze_expert_activation,
    evaluate_moe_gasp,
    compare_moe_vs_standard,
)

from gasp.moe.training_data import (
    sample_tissue_parameters,
    generate_ssfp_training_data,
    generate_stratified_training_data,
    partition_by_t2_t1,
)

from gasp.moe.gating import (
    GatingNetwork,
    FeatureGating,
    MLPGating,
    TemplateGating,
    TopKGating,
    create_gating_network,
)

__all__ = [
    # Main classes and functions
    'MoEGASPConfig',
    'MoEGASPModel',
    'train_moe_gasp',
    'run_moe_gasp',
    'train_moe_gasp_from_image',
    'train_moe_gasp_with_coils',
    'analyze_expert_activation',
    'evaluate_moe_gasp',
    'compare_moe_vs_standard',
    # Training data generation
    'sample_tissue_parameters',
    'generate_ssfp_training_data',
    'generate_stratified_training_data',
    'partition_by_t2_t1',
    # Gating networks
    'GatingNetwork',
    'FeatureGating',
    'MLPGating',
    'TemplateGating',
    'TopKGating',
    'create_gating_network',
]
