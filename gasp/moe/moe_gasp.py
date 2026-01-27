"""Mixture of Experts GASP for heterogeneous T2/T1 data.

This module implements MoE-GASP, which trains K specialized expert GASP models
optimized for different T2/T1 regimes. A gating mechanism learns to weight
expert contributions based on each voxel's signal characteristics.

Mathematical formulation:
    Standard GASP:  output = Phi(x) @ A
    MoE-GASP:       output = sum_k w_k(x) * [Phi(x) @ A_k]

    where:
        A_k = expert coefficients for regime k
        w_k(x) = gating weight for expert k given signal x
        sum_k w_k(x) = 1 (softmax normalization)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

from gasp.gasp import _to_matrix, _design_matrix, l2_regularization
from gasp.moe.gating import GatingNetwork, create_gating_network
from gasp.moe.training_data import partition_by_t2_t1


@dataclass
class MoEGASPConfig:
    """Configuration for MoE-GASP.

    Parameters
    ----------
    n_experts : int
        Number of expert models to train.
    method : str
        Design matrix method: 'linear', 'affine', 'quad', or 'quad-cross'.
    gating_type : str
        Type of gating network: 'feature', 'mlp', 'template', or 'topk'.
    gating_temperature : float
        Softmax temperature for gating. Lower values produce sharper
        expert selection.
    useL2 : bool
        Whether to use L2 regularization for expert training.
    lam : float
        L2 regularization strength.
    penalise_bias : bool
        Whether to penalize the bias term in regularization.
    """
    n_experts: int = 5
    method: str = 'quad-cross'
    gating_type: Literal['feature', 'mlp', 'template', 'topk'] = 'mlp'
    gating_temperature: float = 1.0
    useL2: bool = True
    lam: float = 1e-2
    penalise_bias: bool = False


@dataclass
class MoEGASPModel:
    """Trained MoE-GASP model.

    Contains all information needed to apply the model to new data.

    Attributes
    ----------
    config : MoEGASPConfig
        Configuration used for training.
    expert_coefficients : list[ndarray]
        Coefficient vectors for each expert.
    gating_network : GatingNetwork
        Trained gating network for expert selection.
    regime_edges : ndarray
        T2/T1 ratio boundaries defining each regime.
    training_info : dict
        Metadata about the training process.
    """
    config: MoEGASPConfig
    expert_coefficients: list[npt.NDArray] = field(default_factory=list)
    gating_network: GatingNetwork | None = None
    regime_edges: npt.NDArray | None = None
    training_info: dict = field(default_factory=dict)


def train_moe_gasp(
    signals: npt.NDArray,
    desired_profile: npt.NDArray,
    t2_t1_ratios: npt.NDArray,
    config: MoEGASPConfig | None = None
) -> MoEGASPModel:
    """
    Train MoE-GASP model on simulated data with known T2/T1 ratios.

    This function trains K expert GASP models, each specialized for a
    different T2/T1 regime, and a gating network that learns to route
    inputs to the appropriate experts.

    Parameters
    ----------
    signals : ndarray [n_samples, n_features]
        Training signals (flattened SSFP data).
    desired_profile : ndarray [n_spectral_points] or [n_samples]
        Desired spectral profile. If 1D with fewer elements than n_samples,
        it will be tiled to match.
    t2_t1_ratios : ndarray [n_samples]
        True T2/T1 ratio for each training sample.
    config : MoEGASPConfig, optional
        Model configuration. Uses defaults if not provided.

    Returns
    -------
    model : MoEGASPModel
        Trained model containing expert coefficients and gating network.

    Examples
    --------
    >>> from gasp.moe import train_moe_gasp, MoEGASPConfig
    >>> from gasp.moe.training_data import generate_ssfp_training_data
    >>> signals, t2_t1, params = generate_ssfp_training_data(5000)
    >>> desired = np.exp(-np.linspace(-2, 2, 16)**2)  # Gaussian profile
    >>> model = train_moe_gasp(signals, desired, t2_t1)
    """
    if config is None:
        config = MoEGASPConfig()

    n_samples = signals.shape[0]

    # Partition training data into regimes
    regime_labels, regime_edges = partition_by_t2_t1(
        t2_t1_ratios,
        n_regimes=config.n_experts,
        method='quantile'
    )

    # Prepare desired profile
    desired_flat = np.asarray(desired_profile).ravel()
    if desired_flat.size < n_samples:
        # Tile profile to match samples
        n_repeats = n_samples // desired_flat.size
        if n_samples % desired_flat.size != 0:
            n_repeats += 1
        desired_tiled = np.tile(desired_flat, n_repeats)[:n_samples]
    else:
        desired_tiled = desired_flat[:n_samples]

    # Train expert GASP model for each regime
    expert_coefficients = []
    samples_per_regime = []

    for k in range(config.n_experts):
        mask = regime_labels == k
        n_regime = mask.sum()
        samples_per_regime.append(int(n_regime))

        if n_regime == 0:
            # No samples in this regime - use zeros
            sample_phi = _design_matrix(signals[:1], config.method)
            expert_coefficients.append(
                np.zeros(sample_phi.shape[1], dtype=signals.dtype)
            )
            continue

        # Get signals for this regime
        regime_signals = signals[mask]
        regime_desired = desired_tiled[mask]

        # Build design matrix and fit
        Phi = _design_matrix(regime_signals, config.method)

        if config.useL2:
            A_k = l2_regularization(
                Phi, regime_desired,
                lam=config.lam,
                penalise_bias=config.penalise_bias
            )
        else:
            A_k = np.linalg.lstsq(Phi, regime_desired, rcond=None)[0]

        expert_coefficients.append(A_k)

    # Train gating network
    gating_network = create_gating_network(
        config.gating_type,
        config.n_experts,
        config.gating_temperature
    )
    gating_network.fit(signals, regime_labels)

    # Create model
    model = MoEGASPModel(
        config=config,
        expert_coefficients=expert_coefficients,
        gating_network=gating_network,
        regime_edges=regime_edges,
        training_info={
            'n_samples': n_samples,
            'samples_per_regime': samples_per_regime,
            't2_t1_range': (float(t2_t1_ratios.min()), float(t2_t1_ratios.max())),
            'n_features': signals.shape[1],
        }
    )

    return model


def run_moe_gasp(
    I: npt.NDArray,
    model: MoEGASPModel
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Apply trained MoE-GASP model to new image data.

    Parameters
    ----------
    I : ndarray [H, W, n_features]
        Input SSFP data.
    model : MoEGASPModel
        Trained MoE-GASP model.

    Returns
    -------
    output : ndarray [H, W]
        GASP output image.
    weights : ndarray [H, W, n_experts]
        Gating weights for each voxel (useful for visualization).

    Examples
    --------
    >>> output, weights = run_moe_gasp(ssfp_data, trained_model)
    >>> dominant_expert = np.argmax(weights, axis=-1)
    """
    X, shape = _to_matrix(I)
    h, w = shape
    n_voxels = X.shape[0]

    # Get gating weights for each voxel
    weights = model.gating_network.predict_weights(X)  # [n_voxels, n_experts]

    # Build design matrix once
    Phi = _design_matrix(X, model.config.method)

    # Compute weighted combination of experts
    output = np.zeros(n_voxels, dtype=Phi.dtype)
    for k, A_k in enumerate(model.expert_coefficients):
        expert_output = Phi @ A_k  # [n_voxels]
        output += weights[:, k] * expert_output

    # Reshape outputs
    output = output.reshape(shape)
    weights = weights.reshape(h, w, model.config.n_experts)

    return output, weights


def train_moe_gasp_from_image(
    I: npt.NDArray,
    D: npt.NDArray,
    t2_t1_map: npt.NDArray,
    config: MoEGASPConfig | None = None
) -> MoEGASPModel:
    """
    Train MoE-GASP directly from image data with known T2/T1 map.

    Useful for training on phantom data where T2/T1 is known per voxel.

    Parameters
    ----------
    I : ndarray [H, W, n_features]
        Input SSFP image data.
    D : ndarray [n_spectral_points]
        Desired spectral profile.
    t2_t1_map : ndarray [H, W]
        T2/T1 ratio for each voxel (e.g., from phantom).
    config : MoEGASPConfig, optional
        Model configuration.

    Returns
    -------
    model : MoEGASPModel
        Trained model.
    """
    X, _ = _to_matrix(I)
    t2_t1_flat = t2_t1_map.flatten()

    # Filter out background (T2/T1 = 0 or NaN)
    valid_mask = (t2_t1_flat > 0) & np.isfinite(t2_t1_flat)
    X_valid = X[valid_mask]
    t2_t1_valid = t2_t1_flat[valid_mask]

    return train_moe_gasp(X_valid, D, t2_t1_valid, config)


def train_moe_gasp_with_coils(
    data: npt.NDArray,
    D: npt.NDArray,
    t2_t1_ratios: npt.NDArray,
    config: MoEGASPConfig | None = None
) -> tuple[npt.NDArray, list[MoEGASPModel]]:
    """
    Train MoE-GASP per coil and combine with RSS.

    Parameters
    ----------
    data : ndarray [H, W, coils, PCs] or [H, W, coils, PCs, TRs]
        Multi-coil SSFP data.
    D : ndarray
        Desired spectral profile.
    t2_t1_ratios : ndarray [n_samples]
        T2/T1 ratios (from simulation).
    config : MoEGASPConfig, optional
        Model configuration.

    Returns
    -------
    rss_output : ndarray [H, W]
        Root-sum-of-squares combined output.
    models : list[MoEGASPModel]
        Per-coil trained models.
    """
    if data.ndim == 4:
        h, w, ncoils, npcs = data.shape
        Xc = data
    elif data.ndim == 5:
        h, w, ncoils, npcs, nTRs = data.shape
        Xc = data.reshape(h, w, ncoils, npcs * nTRs)
    else:
        raise ValueError("Expected 4D or 5D data")

    outputs = np.zeros((ncoils, h, w), dtype=complex)
    models = []

    # Prepare training data tiled for all voxels
    n_voxels = h * w

    for c in range(ncoils):
        coil_data = Xc[:, :, c, :]
        X, _ = _to_matrix(coil_data)

        # Train model for this coil
        model_c = train_moe_gasp(X, D, t2_t1_ratios, config)
        models.append(model_c)

        # Apply model
        out_c, _ = run_moe_gasp(coil_data, model_c)
        outputs[c] = out_c

    # RSS combination
    rss = np.sqrt(np.sum(np.abs(outputs) ** 2, axis=0))

    return rss, models


def analyze_expert_activation(
    I: npt.NDArray,
    model: MoEGASPModel,
    t2_t1_map: npt.NDArray | None = None
) -> dict:
    """
    Analyze which experts are activated for different regions.

    This is useful for understanding how the model routes different
    tissue types to different experts, and for validating that the
    gating network is learning meaningful patterns.

    Parameters
    ----------
    I : ndarray [H, W, n_features]
        Input SSFP data.
    model : MoEGASPModel
        Trained MoE-GASP model.
    t2_t1_map : ndarray [H, W], optional
        Ground truth T2/T1 map for accuracy computation.

    Returns
    -------
    dict
        Analysis results with keys:
        - 'weights': [H, W, n_experts] gating weights
        - 'dominant_expert': [H, W] index of highest-weight expert
        - 'expert_entropy': [H, W] entropy of weight distribution
        - 'regime_accuracy': float (if t2_t1_map provided)
    """
    _, weights = run_moe_gasp(I, model)

    dominant = np.argmax(weights, axis=-1)

    # Entropy: -sum(p * log(p))
    # Higher entropy = more uncertainty about which expert to use
    entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)

    result = {
        'weights': weights,
        'dominant_expert': dominant,
        'expert_entropy': entropy
    }

    # If ground truth T2/T1 is available, compute accuracy
    if t2_t1_map is not None:
        true_labels, _ = partition_by_t2_t1(
            t2_t1_map.flatten(),
            n_regimes=model.config.n_experts
        )
        true_labels = true_labels.reshape(t2_t1_map.shape)

        valid_mask = t2_t1_map > 0
        if valid_mask.sum() > 0:
            accuracy = (dominant[valid_mask] == true_labels[valid_mask]).mean()
            result['regime_accuracy'] = float(accuracy)

    return result


def evaluate_moe_gasp(
    model: MoEGASPModel,
    test_signals: npt.NDArray,
    test_t2_t1: npt.NDArray,
    desired_profile: npt.NDArray
) -> dict:
    """
    Evaluate MoE-GASP on test data.

    Parameters
    ----------
    model : MoEGASPModel
        Trained model to evaluate.
    test_signals : ndarray [n_samples, n_features]
        Test signals.
    test_t2_t1 : ndarray [n_samples]
        True T2/T1 ratios for test samples.
    desired_profile : ndarray
        Desired output profile.

    Returns
    -------
    dict
        Evaluation metrics including:
        - 'mse': Overall mean squared error
        - 'regime_mse': Dict mapping regime index to MSE
        - 'gating_entropy_mean': Mean gating entropy
    """
    n = test_signals.shape[0]
    side = int(np.sqrt(n))
    if side * side > n:
        side -= 1

    # Reshape to image for run_moe_gasp
    I_test = test_signals[:side * side].reshape(side, side, -1)

    output, weights = run_moe_gasp(I_test, model)
    output_flat = output.flatten()

    # Tile desired profile
    desired = np.asarray(desired_profile).ravel()
    desired_tiled = np.tile(desired, side * side // desired.size + 1)[:side * side]

    # Overall MSE
    mse = np.mean(np.abs(output_flat - desired_tiled) ** 2)

    # Per-regime MSE
    regime_labels, _ = partition_by_t2_t1(
        test_t2_t1[:side * side],
        n_regimes=model.config.n_experts
    )

    regime_mse = {}
    for k in range(model.config.n_experts):
        mask = regime_labels == k
        if mask.sum() > 0:
            regime_mse[k] = float(
                np.mean(np.abs(output_flat[mask] - desired_tiled[mask]) ** 2)
            )

    # Gating entropy
    entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)

    return {
        'mse': float(mse),
        'regime_mse': regime_mse,
        'gating_entropy_mean': float(entropy.mean())
    }


def compare_moe_vs_standard(
    test_data: npt.NDArray,
    desired_profile: npt.NDArray,
    moe_model: MoEGASPModel,
    standard_coeffs: npt.NDArray,
    method: str = 'quad-cross'
) -> dict:
    """
    Compare MoE-GASP to standard single-model GASP.

    Parameters
    ----------
    test_data : ndarray [H, W, n_features]
        Test image data.
    desired_profile : ndarray
        Desired spectral profile.
    moe_model : MoEGASPModel
        Trained MoE-GASP model.
    standard_coeffs : ndarray
        Coefficients from standard GASP training.
    method : str
        Method used for standard GASP.

    Returns
    -------
    dict
        Comparison metrics:
        - 'standard_mse': MSE for standard GASP
        - 'moe_mse': MSE for MoE-GASP
        - 'improvement': Percent improvement
    """
    from gasp.gasp import run_gasp

    # Standard GASP output
    standard_output = run_gasp(test_data, standard_coeffs, method)

    # MoE-GASP output
    moe_output, _ = run_moe_gasp(test_data, moe_model)

    # Compute metrics
    h, w = test_data.shape[:2]
    desired = np.asarray(desired_profile).ravel()
    desired_tiled = np.tile(desired, h * w // desired.size + 1)[:h * w].reshape(h, w)

    standard_mse = np.mean(np.abs(standard_output - desired_tiled) ** 2)
    moe_mse = np.mean(np.abs(moe_output - desired_tiled) ** 2)

    improvement = (standard_mse - moe_mse) / (standard_mse + 1e-10) * 100

    return {
        'standard_mse': float(standard_mse),
        'moe_mse': float(moe_mse),
        'improvement': float(improvement)
    }
