# Mixture of Experts GASP: Detailed Implementation Plan

## 1. Overview

### Motivation
The standard GASP model learns a single set of coefficients `A` that maps signal features to a desired spectral profile. This works well when tissues have similar T2/T1 ratios, but fails for heterogeneous data where different tissues produce fundamentally different SSFP signal shapes.

### Solution: Mixture of Experts (MoE)
Instead of one global model, train K specialized "expert" GASP models, each optimized for a specific T2/T1 regime. A gating mechanism learns to weight expert contributions based on each voxel's signal characteristics.

### Mathematical Formulation

**Standard GASP:**
```
output = Φ(x) · A
```

**MoE-GASP:**
```
output = Σ_k  w_k(x) · [Φ(x) · A_k]

where:
  - A_k = expert coefficients for regime k
  - w_k(x) = gating weight for expert k given signal x
  - Σ_k w_k(x) = 1  (softmax normalization)
```

---

## 2. Architecture Design

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      MoE-GASP Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Signal x [n_features]                                    │
│         │                                                       │
│         ├──────────────────┬────────────────────────┐          │
│         │                  │                        │          │
│         ▼                  ▼                        ▼          │
│  ┌─────────────┐    ┌─────────────┐         ┌─────────────┐   │
│  │  Expert 1   │    │  Expert 2   │   ...   │  Expert K   │   │
│  │ (low T2/T1) │    │ (mid T2/T1) │         │(high T2/T1) │   │
│  │  Φ · A_1    │    │  Φ · A_2    │         │  Φ · A_K    │   │
│  └──────┬──────┘    └──────┬──────┘         └──────┬──────┘   │
│         │                  │                        │          │
│         │                  │                        │          │
│         ▼                  ▼                        ▼          │
│       y_1                y_2                      y_K          │
│         │                  │                        │          │
│         └──────────────────┼────────────────────────┘          │
│                            │                                    │
│                            ▼                                    │
│  Input Signal x ──► ┌─────────────┐                            │
│                     │   Gating    │ ──► [w_1, w_2, ..., w_K]   │
│                     │   Network   │      (softmax weights)      │
│                     └─────────────┘                            │
│                            │                                    │
│                            ▼                                    │
│                   output = Σ w_k · y_k                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Expert Design

Each expert is a standard GASP model with its own coefficient vector:

| Expert | T2/T1 Range | Example Tissues |
|--------|-------------|-----------------|
| 1 | 0.001 - 0.02 | Proteins, tendons |
| 2 | 0.02 - 0.06 | Liver, muscle |
| 3 | 0.06 - 0.12 | White matter, fat |
| 4 | 0.12 - 0.25 | Gray matter |
| 5 | 0.25 - 0.50 | Fluids, water |

### 2.3 Gating Network Options

**Option A: Feature-Based Gating (Simple)**
```python
# Extract discriminative features from signal
features = [
    mean(|x|),           # Signal magnitude
    std(|x|),            # Magnitude variation
    mean(angle(x)),      # Mean phase
    |x_max| / |x_min|,   # Dynamic range
    spectral_centroid,   # Center of mass in spectral dim
]
# Linear classifier + softmax
weights = softmax(W @ features + b)
```

**Option B: MLP Gating (Flexible)**
```python
# Small neural network
hidden = ReLU(W1 @ x + b1)
weights = softmax(W2 @ hidden + b2)
```

**Option C: Signal-Matched Gating (Physics-Informed)**
```python
# Compare signal to template signals for each regime
templates = [mean_signal_for_regime_k for k in range(K)]
similarities = [cosine_sim(x, t) for t in templates]
weights = softmax(similarities / temperature)
```

---

## 3. Implementation Plan

### 3.1 File Structure

```
gasp/
├── gasp.py              # Existing GASP (unchanged)
├── ssfp.py              # Existing SSFP model (unchanged)
├── tissue.py            # Existing tissue maps (unchanged)
├── moe_gasp.py          # NEW: Mixture of Experts GASP
├── gating.py            # NEW: Gating network implementations
├── training_data.py     # NEW: Synthetic data generation
└── __init__.py          # Update exports
```

### 3.2 Module: `training_data.py`

Generate diverse training data with known T2/T1 ratios.

```python
"""Training data generation for MoE-GASP."""

import numpy as np
from gasp.ssfp import ssfp, add_noise


def sample_tissue_parameters(
    n_samples: int,
    t2_t1_range: tuple[float, float] = (0.001, 0.5),
    t1_range: tuple[float, float] = (0.2, 2.0),
    f0_range: tuple[float, float] = (-200, 200),
    seed: int | None = None
) -> dict[str, np.ndarray]:
    """
    Sample tissue parameters covering the T2/T1 space.

    Returns dict with keys: 't1', 't2', 't2_t1_ratio', 'f0'
    """
    rng = np.random.default_rng(seed)

    # Log-uniform sampling for T2/T1 (better coverage of short T2)
    log_min, log_max = np.log(t2_t1_range[0]), np.log(t2_t1_range[1])
    t2_t1_ratios = np.exp(rng.uniform(log_min, log_max, n_samples))

    # Uniform T1 sampling
    t1_values = rng.uniform(t1_range[0], t1_range[1], n_samples)
    t2_values = t2_t1_ratios * t1_values

    # Clamp T2 to physiological range
    t2_values = np.clip(t2_values, 0.001, 2.0)

    # Off-resonance sampling
    f0_values = rng.uniform(f0_range[0], f0_range[1], n_samples)

    return {
        't1': t1_values,
        't2': t2_values,
        't2_t1_ratio': t2_t1_ratios,
        'f0': f0_values
    }


def generate_ssfp_training_data(
    n_samples: int,
    TR: float | list[float] = 0.005,
    TE: float | None = None,
    alpha: float = np.deg2rad(30),
    npcs: int = 16,
    t2_t1_range: tuple[float, float] = (0.001, 0.5),
    add_noise_flag: bool = True,
    noise_sigma: float = 0.01,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate SSFP signals for training MoE-GASP.

    Parameters
    ----------
    n_samples : int
        Number of training samples to generate
    TR : float or list[float]
        Repetition time(s) in seconds
    TE : float, optional
        Echo time (defaults to TR/2)
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles
    t2_t1_range : tuple
        Range of T2/T1 ratios to sample
    add_noise_flag : bool
        Whether to add Gaussian noise
    noise_sigma : float
        Noise standard deviation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    signals : ndarray [n_samples, n_features]
        SSFP signals (flattened across phase cycles and TRs)
    t2_t1_ratios : ndarray [n_samples]
        True T2/T1 ratio for each sample
    params : dict
        Full parameter dictionary
    """
    TRs = np.atleast_1d(TR)
    TE = TE if TE is not None else TRs[0] / 2

    # Sample tissue parameters
    params = sample_tissue_parameters(
        n_samples,
        t2_t1_range=t2_t1_range,
        seed=seed
    )

    # Phase cycle increments
    dphi = np.linspace(0, 2 * np.pi, npcs, endpoint=False)

    # Generate signals
    signals = []
    for i in range(n_samples):
        sample_signals = []
        for tr in TRs:
            te = tr / 2
            sig = ssfp(
                T1=params['t1'][i],
                T2=params['t2'][i],
                TR=tr,
                TE=te,
                alpha=alpha,
                dphi=dphi,
                f0=params['f0'][i],
                field_map=0,
                M0=1.0
            )
            if add_noise_flag:
                sig = add_noise(sig, sigma=noise_sigma)
            sample_signals.append(sig.flatten())

        signals.append(np.concatenate(sample_signals))

    signals = np.array(signals)

    return signals, params['t2_t1_ratio'], params


def partition_by_t2_t1(
    t2_t1_ratios: np.ndarray,
    n_regimes: int = 5,
    method: str = 'quantile'
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition samples into T2/T1 regimes.

    Parameters
    ----------
    t2_t1_ratios : ndarray
        T2/T1 ratios for each sample
    n_regimes : int
        Number of regimes/experts
    method : str
        'quantile' for equal-count bins, 'uniform' for equal-width bins

    Returns
    -------
    regime_labels : ndarray [n_samples]
        Regime index (0 to n_regimes-1) for each sample
    regime_edges : ndarray [n_regimes + 1]
        Bin edges defining regime boundaries
    """
    if method == 'quantile':
        # Equal number of samples per regime
        percentiles = np.linspace(0, 100, n_regimes + 1)
        regime_edges = np.percentile(t2_t1_ratios, percentiles)
    elif method == 'uniform':
        # Equal-width bins in log space
        log_ratios = np.log(t2_t1_ratios)
        regime_edges = np.exp(np.linspace(log_ratios.min(), log_ratios.max(), n_regimes + 1))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Assign labels
    regime_labels = np.digitize(t2_t1_ratios, regime_edges[1:-1])

    return regime_labels, regime_edges
```

### 3.3 Module: `gating.py`

Gating network implementations.

```python
"""Gating networks for MoE-GASP."""

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class GatingNetwork(ABC):
    """Abstract base class for gating networks."""

    @abstractmethod
    def fit(self, signals: np.ndarray, regime_labels: np.ndarray) -> None:
        """Train the gating network."""
        pass

    @abstractmethod
    def predict_weights(self, signals: np.ndarray) -> np.ndarray:
        """Predict soft weights for each expert."""
        pass


class FeatureGating(GatingNetwork):
    """
    Feature-based gating using handcrafted signal features.
    Uses logistic regression for classification.
    """

    def __init__(self, n_experts: int, temperature: float = 1.0):
        self.n_experts = n_experts
        self.temperature = temperature
        self.weights = None
        self.bias = None

    def _extract_features(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract discriminative features from signals.

        signals: [n_samples, n_features] complex
        returns: [n_samples, n_extracted_features] real
        """
        mag = np.abs(signals)
        phase = np.angle(signals)

        features = np.column_stack([
            mag.mean(axis=1),                    # Mean magnitude
            mag.std(axis=1),                     # Magnitude std
            mag.max(axis=1) / (mag.min(axis=1) + 1e-10),  # Dynamic range
            phase.mean(axis=1),                  # Mean phase
            phase.std(axis=1),                   # Phase std
            np.abs(np.fft.fft(mag, axis=1)[:, 1]),  # First spectral component
        ])
        return features

    def fit(self, signals: np.ndarray, regime_labels: np.ndarray) -> None:
        """Train gating using softmax regression."""
        features = self._extract_features(signals)
        n_samples, n_features = features.shape

        # One-hot encode labels
        one_hot = np.zeros((n_samples, self.n_experts))
        one_hot[np.arange(n_samples), regime_labels] = 1

        # Initialize weights
        self.weights = np.zeros((n_features, self.n_experts))
        self.bias = np.zeros(self.n_experts)

        # Simple gradient descent (could use scipy.optimize)
        lr = 0.01
        for _ in range(1000):
            logits = features @ self.weights + self.bias
            probs = self._softmax(logits)

            # Gradient
            grad_logits = (probs - one_hot) / n_samples
            grad_w = features.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted / self.temperature)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_weights(self, signals: np.ndarray) -> np.ndarray:
        """Predict soft weights for each expert."""
        features = self._extract_features(signals)
        logits = features @ self.weights + self.bias
        return self._softmax(logits)


class MLPGating(GatingNetwork):
    """
    MLP-based gating network.
    Uses a small neural network for more flexible gating.
    """

    def __init__(
        self,
        n_experts: int,
        hidden_dim: int = 32,
        temperature: float = 1.0
    ):
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def fit(self, signals: np.ndarray, regime_labels: np.ndarray) -> None:
        """Train MLP gating network."""
        # Use magnitude of complex signals as input
        X = np.abs(signals)
        n_samples, input_dim = X.shape

        # One-hot encode labels
        one_hot = np.zeros((n_samples, self.n_experts))
        one_hot[np.arange(n_samples), regime_labels] = 1

        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_dim, self.hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.n_experts) * np.sqrt(2 / self.hidden_dim)
        self.b2 = np.zeros(self.n_experts)

        # Training loop
        lr = 0.001
        for epoch in range(2000):
            # Forward pass
            hidden = self._relu(X @ self.W1 + self.b1)
            logits = hidden @ self.W2 + self.b2
            probs = self._softmax(logits)

            # Backward pass
            grad_logits = (probs - one_hot) / n_samples
            grad_W2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)

            grad_hidden = grad_logits @ self.W2.T
            grad_hidden[hidden <= 0] = 0  # ReLU derivative
            grad_W1 = X.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            # Update
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted / self.temperature)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict_weights(self, signals: np.ndarray) -> np.ndarray:
        """Predict soft weights for each expert."""
        X = np.abs(signals)
        hidden = self._relu(X @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return self._softmax(logits)


class TemplateGating(GatingNetwork):
    """
    Template-matching gating using prototype signals.
    Physics-informed approach using mean signals per regime.
    """

    def __init__(self, n_experts: int, temperature: float = 0.1):
        self.n_experts = n_experts
        self.temperature = temperature
        self.templates = None  # [n_experts, n_features]

    def fit(self, signals: np.ndarray, regime_labels: np.ndarray) -> None:
        """Compute template (mean signal) for each regime."""
        self.templates = np.zeros((self.n_experts, signals.shape[1]), dtype=signals.dtype)

        for k in range(self.n_experts):
            mask = regime_labels == k
            if mask.sum() > 0:
                self.templates[k] = signals[mask].mean(axis=0)

    def predict_weights(self, signals: np.ndarray) -> np.ndarray:
        """Predict weights based on cosine similarity to templates."""
        # Normalize signals and templates
        sig_norm = signals / (np.linalg.norm(signals, axis=1, keepdims=True) + 1e-10)
        temp_norm = self.templates / (np.linalg.norm(self.templates, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity: [n_samples, n_experts]
        similarities = np.real(sig_norm @ temp_norm.conj().T)

        # Softmax with temperature
        shifted = similarities - similarities.max(axis=1, keepdims=True)
        exp_sim = np.exp(shifted / self.temperature)
        return exp_sim / exp_sim.sum(axis=1, keepdims=True)
```

### 3.4 Module: `moe_gasp.py`

Main MoE-GASP implementation.

```python
"""Mixture of Experts GASP for heterogeneous T2/T1 data."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field

from gasp.gasp import train_gasp, run_gasp, _to_matrix, _design_matrix
from gasp.gating import GatingNetwork, FeatureGating, MLPGating, TemplateGating
from gasp.training_data import (
    generate_ssfp_training_data,
    partition_by_t2_t1
)


@dataclass
class MoEGASPConfig:
    """Configuration for MoE-GASP."""
    n_experts: int = 5
    method: str = 'quad-cross'
    gating_type: str = 'mlp'  # 'feature', 'mlp', or 'template'
    gating_temperature: float = 1.0
    useL2: bool = True
    lam: float = 1e-2
    penalise_bias: bool = False


@dataclass
class MoEGASPModel:
    """Trained MoE-GASP model."""
    config: MoEGASPConfig
    expert_coefficients: list[npt.NDArray] = field(default_factory=list)
    gating_network: GatingNetwork | None = None
    regime_edges: npt.NDArray | None = None
    training_info: dict = field(default_factory=dict)


def create_gating_network(
    gating_type: str,
    n_experts: int,
    temperature: float = 1.0
) -> GatingNetwork:
    """Factory function for gating networks."""
    if gating_type == 'feature':
        return FeatureGating(n_experts, temperature)
    elif gating_type == 'mlp':
        return MLPGating(n_experts, temperature=temperature)
    elif gating_type == 'template':
        return TemplateGating(n_experts, temperature)
    else:
        raise ValueError(f"Unknown gating type: {gating_type}")


def train_moe_gasp(
    signals: npt.NDArray,
    desired_profile: npt.NDArray,
    t2_t1_ratios: npt.NDArray,
    config: MoEGASPConfig | None = None
) -> MoEGASPModel:
    """
    Train MoE-GASP model on simulated data with known T2/T1 ratios.

    Parameters
    ----------
    signals : ndarray [n_samples, n_features]
        Training signals (flattened SSFP data)
    desired_profile : ndarray [n_samples] or [n_spectral_points]
        Desired spectral profile for each sample or single profile to tile
    t2_t1_ratios : ndarray [n_samples]
        True T2/T1 ratio for each training sample
    config : MoEGASPConfig, optional
        Model configuration

    Returns
    -------
    model : MoEGASPModel
        Trained model containing expert coefficients and gating network
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

    # Train expert GASP model for each regime
    expert_coefficients = []
    for k in range(config.n_experts):
        mask = regime_labels == k
        n_regime = mask.sum()

        if n_regime == 0:
            # No samples in this regime - use zeros
            # Determine coefficient size from design matrix
            sample_phi = _design_matrix(signals[:1], config.method)
            expert_coefficients.append(np.zeros(sample_phi.shape[1], dtype=signals.dtype))
            continue

        # Get signals for this regime
        regime_signals = signals[mask]

        # Handle desired profile
        if desired_profile.ndim == 1 and desired_profile.size != n_regime:
            # Single profile to tile
            regime_desired = np.tile(desired_profile, n_regime // desired_profile.size + 1)[:n_regime]
        elif desired_profile.ndim == 1:
            regime_desired = desired_profile[mask] if desired_profile.size == n_samples else desired_profile
        else:
            regime_desired = desired_profile[mask]

        # Build design matrix and fit
        Phi = _design_matrix(regime_signals, config.method)

        if config.useL2:
            from gasp.gasp import l2_regularization
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
            'samples_per_regime': [int((regime_labels == k).sum()) for k in range(config.n_experts)],
            't2_t1_range': (t2_t1_ratios.min(), t2_t1_ratios.max())
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
        Input SSFP data
    model : MoEGASPModel
        Trained MoE-GASP model

    Returns
    -------
    output : ndarray [H, W]
        GASP output image
    weights : ndarray [H, W, n_experts]
        Gating weights for each voxel (useful for visualization)
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
        Input SSFP image data
    D : ndarray [n_spectral_points]
        Desired spectral profile
    t2_t1_map : ndarray [H, W]
        T2/T1 ratio for each voxel (from phantom)
    config : MoEGASPConfig, optional
        Model configuration

    Returns
    -------
    model : MoEGASPModel
        Trained model
    """
    X, shape = _to_matrix(I)
    t2_t1_flat = t2_t1_map.flatten()

    # Filter out background (T2/T1 = 0 or NaN)
    valid_mask = (t2_t1_flat > 0) & np.isfinite(t2_t1_flat)
    X_valid = X[valid_mask]
    t2_t1_valid = t2_t1_flat[valid_mask]

    # Tile desired profile
    D_tiled = np.tile(D, X_valid.shape[0] // D.size + 1)[:X_valid.shape[0]]

    return train_moe_gasp(X_valid, D_tiled, t2_t1_valid, config)


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
        Multi-coil SSFP data
    D : ndarray
        Desired spectral profile
    t2_t1_ratios : ndarray [n_samples]
        T2/T1 ratios (from simulation)
    config : MoEGASPConfig, optional
        Model configuration

    Returns
    -------
    rss_output : ndarray [H, W]
        Root-sum-of-squares combined output
    models : list[MoEGASPModel]
        Per-coil trained models
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

    for c in range(ncoils):
        coil_data = Xc[:, :, c, :]
        X, _ = _to_matrix(coil_data)

        # Train model for this coil
        D_tiled = np.tile(D, X.shape[0] // D.size + 1)[:X.shape[0]]
        model_c = train_moe_gasp(X, D_tiled, t2_t1_ratios, config)
        models.append(model_c)

        # Apply model
        out_c, _ = run_moe_gasp(coil_data, model_c)
        outputs[c] = out_c

    # RSS combination
    rss = np.sqrt(np.sum(np.abs(outputs) ** 2, axis=0))

    return rss, models


# ---------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------

def analyze_expert_activation(
    I: npt.NDArray,
    model: MoEGASPModel,
    t2_t1_map: npt.NDArray | None = None
) -> dict:
    """
    Analyze which experts are activated for different regions.

    Returns
    -------
    dict with keys:
        - 'weights': [H, W, n_experts] gating weights
        - 'dominant_expert': [H, W] index of highest-weight expert
        - 'expert_entropy': [H, W] entropy of weight distribution (uncertainty)
        - 'regime_accuracy': float (if t2_t1_map provided)
    """
    _, weights = run_moe_gasp(I, model)

    dominant = np.argmax(weights, axis=-1)

    # Entropy: -sum(p * log(p))
    entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)

    result = {
        'weights': weights,
        'dominant_expert': dominant,
        'expert_entropy': entropy
    }

    # If ground truth T2/T1 is available, compute accuracy
    if t2_t1_map is not None:
        from gasp.training_data import partition_by_t2_t1
        true_labels, _ = partition_by_t2_t1(
            t2_t1_map.flatten(),
            n_regimes=model.config.n_experts
        )
        true_labels = true_labels.reshape(t2_t1_map.shape)

        valid_mask = t2_t1_map > 0
        accuracy = (dominant[valid_mask] == true_labels[valid_mask]).mean()
        result['regime_accuracy'] = accuracy

    return result
```

---

## 4. Training Procedure

### 4.1 Step-by-Step Training

```python
# Example training script

import numpy as np
from gasp.moe_gasp import train_moe_gasp, MoEGASPConfig
from gasp.training_data import generate_ssfp_training_data
from gasp.responses import gaussian_response

# 1. Generate training data
print("Generating training data...")
signals, t2_t1_ratios, params = generate_ssfp_training_data(
    n_samples=10000,
    TR=[0.005, 0.010, 0.020],  # Multiple TRs
    alpha=np.deg2rad(30),
    npcs=16,
    t2_t1_range=(0.001, 0.5),
    add_noise_flag=True,
    noise_sigma=0.01,
    seed=42
)
print(f"Generated {signals.shape[0]} samples with {signals.shape[1]} features each")

# 2. Create desired spectral profile
npcs = 16
desired_profile = gaussian_response(npcs, center=npcs//2, sigma=2)

# 3. Configure MoE-GASP
config = MoEGASPConfig(
    n_experts=5,
    method='quad-cross',
    gating_type='mlp',
    gating_temperature=1.0,
    useL2=True,
    lam=1e-2
)

# 4. Train model
print("Training MoE-GASP...")
model = train_moe_gasp(signals, desired_profile, t2_t1_ratios, config)

print(f"Trained {config.n_experts} experts")
print(f"Samples per regime: {model.training_info['samples_per_regime']}")
print(f"T2/T1 range: {model.training_info['t2_t1_range']}")

# 5. Save model (optional)
import pickle
with open('moe_gasp_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### 4.2 Hyperparameter Tuning

| Parameter | Range to Try | Notes |
|-----------|--------------|-------|
| `n_experts` | 3, 5, 7, 10 | More experts = finer T2/T1 resolution, but more data needed |
| `method` | 'affine', 'quad', 'quad-cross' | Higher order = more expressive, risk of overfitting |
| `gating_type` | 'feature', 'mlp', 'template' | Start with 'template', try 'mlp' for more flexibility |
| `gating_temperature` | 0.1, 0.5, 1.0, 2.0 | Lower = sharper expert selection; higher = smoother blending |
| `lam` | 1e-4, 1e-3, 1e-2, 1e-1 | Regularization strength |
| `n_samples` | 5000, 10000, 50000 | More samples = better generalization |

---

## 5. Inference Procedure

### 5.1 Apply to New Data

```python
from gasp.moe_gasp import run_moe_gasp, analyze_expert_activation
import pickle

# Load trained model
with open('moe_gasp_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new SSFP data: [H, W, n_features]
# new_data = load_your_data(...)

# Apply MoE-GASP
output, weights = run_moe_gasp(new_data, model)

# Analyze expert activation
analysis = analyze_expert_activation(new_data, model)
dominant_expert = analysis['dominant_expert']
uncertainty = analysis['expert_entropy']

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(np.abs(output), cmap='gray')
axes[0].set_title('MoE-GASP Output')

axes[1].imshow(dominant_expert, cmap='tab10', vmin=0, vmax=model.config.n_experts-1)
axes[1].set_title('Dominant Expert')

axes[2].imshow(uncertainty, cmap='hot')
axes[2].set_title('Gating Uncertainty (Entropy)')

# Show individual expert weights
for k in range(min(model.config.n_experts, 5)):
    axes[3].plot(weights[128, :, k], label=f'Expert {k}')
axes[3].legend()
axes[3].set_title('Expert Weights (row 128)')

plt.tight_layout()
plt.show()
```

---

## 6. Evaluation Strategy

### 6.1 Simulation Metrics

```python
def evaluate_moe_gasp(
    model: MoEGASPModel,
    test_signals: np.ndarray,
    test_t2_t1: np.ndarray,
    desired_profile: np.ndarray
) -> dict:
    """Evaluate MoE-GASP on test data."""

    # Get outputs
    # Reshape signals to [sqrt(n), sqrt(n), features] for run_moe_gasp
    n = test_signals.shape[0]
    side = int(np.sqrt(n))
    I_test = test_signals[:side*side].reshape(side, side, -1)

    output, weights = run_moe_gasp(I_test, model)
    output_flat = output.flatten()

    # Tile desired profile
    desired = np.tile(desired_profile, side*side // desired_profile.size)

    # MSE
    mse = np.mean(np.abs(output_flat - desired) ** 2)

    # Per-regime MSE
    from gasp.training_data import partition_by_t2_t1
    regime_labels, _ = partition_by_t2_t1(test_t2_t1[:side*side], model.config.n_experts)

    regime_mse = {}
    for k in range(model.config.n_experts):
        mask = regime_labels == k
        if mask.sum() > 0:
            regime_mse[k] = np.mean(np.abs(output_flat[mask] - desired[mask]) ** 2)

    return {
        'mse': mse,
        'regime_mse': regime_mse,
        'gating_entropy_mean': -np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()
    }
```

### 6.2 Comparison with Standard GASP

```python
def compare_moe_vs_standard(
    test_data: np.ndarray,
    desired_profile: np.ndarray,
    moe_model: MoEGASPModel,
    standard_coeffs: np.ndarray,
    method: str = 'quad-cross'
) -> dict:
    """Compare MoE-GASP to standard single-model GASP."""

    from gasp.gasp import run_gasp

    # Standard GASP output
    standard_output = run_gasp(test_data, standard_coeffs, method)

    # MoE-GASP output
    moe_output, _ = run_moe_gasp(test_data, moe_model)

    # Compute metrics
    h, w = test_data.shape[:2]
    desired_tiled = np.tile(desired_profile, h * w // desired_profile.size).reshape(h, w)

    standard_mse = np.mean(np.abs(standard_output - desired_tiled) ** 2)
    moe_mse = np.mean(np.abs(moe_output - desired_tiled) ** 2)

    return {
        'standard_mse': standard_mse,
        'moe_mse': moe_mse,
        'improvement': (standard_mse - moe_mse) / standard_mse * 100  # Percent improvement
    }
```

---

## 7. Integration Checklist

### Phase 1: Core Implementation
- [ ] Create `gasp/training_data.py` with data generation functions
- [ ] Create `gasp/gating.py` with gating network classes
- [ ] Create `gasp/moe_gasp.py` with main MoE-GASP class
- [ ] Update `gasp/__init__.py` to export new modules
- [ ] Write unit tests for each module

### Phase 2: Validation
- [ ] Test on synthetic heterogeneous phantom
- [ ] Compare MoE-GASP vs standard GASP on multi-tissue data
- [ ] Tune hyperparameters (n_experts, gating_type, temperature)
- [ ] Visualize expert activations

### Phase 3: Real Data
- [ ] Test on existing datasets (knee, brain, etc.)
- [ ] Evaluate on data with known tissue boundaries
- [ ] Document failure cases and limitations

### Phase 4: Extensions
- [ ] Add support for end-to-end fine-tuning
- [ ] Implement sparse MoE (top-k experts only)
- [ ] Add uncertainty quantification

---

## 8. Expected Outcomes

### Improvements over Standard GASP
1. **Better fitting for heterogeneous tissues**: Each expert specializes in its T2/T1 regime
2. **Interpretable segmentation**: Dominant expert map shows tissue-like boundaries
3. **Uncertainty quantification**: Gating entropy indicates confidence

### Potential Limitations
1. **Requires simulation-to-real transfer**: Model trained on simulated data
2. **More parameters**: K experts × n_coefficients
3. **Gating errors**: Wrong expert selection degrades performance

### Mitigation Strategies
1. Add domain randomization (noise, B0 inhomogeneity) during training
2. Use regularization and proper validation
3. Use soft gating (blending) rather than hard expert selection
