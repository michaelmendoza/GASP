"""Gating networks for MoE-GASP.

This module implements various gating network architectures that learn to
route input signals to the appropriate expert based on signal characteristics.
The gating network produces soft weights that determine how much each expert
contributes to the final output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class GatingNetwork(ABC):
    """Abstract base class for gating networks.

    A gating network learns to predict which expert(s) should handle a given
    input signal. It produces soft weights (summing to 1) that blend the
    outputs of multiple experts.
    """

    @abstractmethod
    def fit(self, signals: npt.NDArray, regime_labels: npt.NDArray) -> None:
        """Train the gating network on labeled data.

        Parameters
        ----------
        signals : ndarray [n_samples, n_features]
            Training signals.
        regime_labels : ndarray [n_samples]
            Expert/regime label for each sample (0 to n_experts-1).
        """
        pass

    @abstractmethod
    def predict_weights(self, signals: npt.NDArray) -> npt.NDArray:
        """Predict soft weights for each expert.

        Parameters
        ----------
        signals : ndarray [n_samples, n_features]
            Input signals.

        Returns
        -------
        weights : ndarray [n_samples, n_experts]
            Soft weights summing to 1 for each sample.
        """
        pass


class FeatureGating(GatingNetwork):
    """Feature-based gating using handcrafted signal features.

    Extracts discriminative features from the input signals (magnitude,
    phase statistics, spectral properties) and uses softmax regression
    to predict expert weights.

    Parameters
    ----------
    n_experts : int
        Number of experts to route between.
    temperature : float
        Softmax temperature. Lower values produce sharper expert selection.
    n_iterations : int
        Number of gradient descent iterations for training.
    learning_rate : float
        Learning rate for gradient descent.
    """

    def __init__(
        self,
        n_experts: int,
        temperature: float = 1.0,
        n_iterations: int = 2000,
        learning_rate: float = 0.1
    ):
        self.n_experts = n_experts
        self.temperature = temperature
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights: npt.NDArray | None = None
        self.bias: npt.NDArray | None = None
        self._feat_mean: npt.NDArray | None = None
        self._feat_std: npt.NDArray | None = None

    def _extract_features(self, signals: npt.NDArray) -> npt.NDArray:
        """Extract discriminative features from signals.

        Parameters
        ----------
        signals : ndarray [n_samples, n_features]
            Complex-valued signals.

        Returns
        -------
        features : ndarray [n_samples, n_extracted_features]
            Real-valued features for classification.
        """
        mag = np.abs(signals)
        phase = np.angle(signals)
        n_samples, n_features = mag.shape

        # Normalize magnitude per sample (remove M0 dependency)
        mag_sum = mag.sum(axis=1, keepdims=True)
        mag_sum = np.where(mag_sum < 1e-10, 1e-10, mag_sum)
        mag_norm = mag / mag_sum

        # Handle edge cases for min values
        mag_min = mag.min(axis=1, keepdims=True)
        mag_min = np.where(mag_min < 1e-10, 1e-10, mag_min)
        mag_max = mag.max(axis=1, keepdims=True)
        mag_max = np.where(mag_max < 1e-10, 1e-10, mag_max)

        # Basic magnitude statistics
        feat_list = [
            mag.mean(axis=1),                           # Mean magnitude
            mag.std(axis=1),                            # Magnitude std
            (mag_max / mag_min).squeeze(),              # Dynamic range
            mag_norm.std(axis=1),                       # Normalized shape variation
        ]

        # Phase statistics (unwrapped for continuity)
        feat_list.extend([
            np.cos(phase).mean(axis=1),                 # Mean cos(phase) - more stable
            np.sin(phase).mean(axis=1),                 # Mean sin(phase)
            phase.std(axis=1),                          # Phase std
        ])

        # Spectral features
        fft_mag = np.abs(np.fft.fft(mag_norm, axis=1))
        feat_list.extend([
            fft_mag[:, 1],                              # First harmonic
            fft_mag[:, 2] if n_features > 2 else np.zeros(n_samples),  # Second harmonic
        ])

        # TR-ratio features (if multiple TRs are present)
        # Assuming features are organized as [PC0_TR0, PC1_TR0, ..., PC0_TR1, PC1_TR1, ...]
        # These ratios capture T2-dependent decay across TRs
        n_pcs = n_features // 3 if n_features >= 6 else n_features
        if n_features >= 2 * n_pcs:
            # Ratio of mean signal between first and second TR blocks
            tr1_mean = mag[:, :n_pcs].mean(axis=1)
            tr2_mean = mag[:, n_pcs:2*n_pcs].mean(axis=1)
            tr1_mean = np.where(tr1_mean < 1e-10, 1e-10, tr1_mean)
            feat_list.append(tr2_mean / tr1_mean)  # Decay ratio

        if n_features >= 3 * n_pcs:
            # Ratio with third TR block
            tr3_mean = mag[:, 2*n_pcs:3*n_pcs].mean(axis=1)
            tr2_mean_safe = np.where(tr2_mean < 1e-10, 1e-10, tr2_mean)
            feat_list.append(tr3_mean / tr2_mean_safe)

        # Higher-order moments of normalized magnitude (shape descriptors)
        mag_centered = mag_norm - mag_norm.mean(axis=1, keepdims=True)
        mag_std = mag_norm.std(axis=1, keepdims=True)
        mag_std = np.where(mag_std < 1e-10, 1e-10, mag_std)
        mag_z = mag_centered / mag_std

        # Skewness and kurtosis
        skewness = (mag_z ** 3).mean(axis=1)
        kurtosis = (mag_z ** 4).mean(axis=1) - 3  # Excess kurtosis
        feat_list.extend([skewness, kurtosis])

        features = np.column_stack(feat_list)

        # Standardize features for better gradient descent
        if not hasattr(self, '_feat_mean') or self._feat_mean is None:
            self._feat_mean = features.mean(axis=0)
            self._feat_std = features.std(axis=0)
            self._feat_std = np.where(self._feat_std < 1e-10, 1.0, self._feat_std)

        features = (features - self._feat_mean) / self._feat_std
        return features

    def _softmax(self, logits: npt.NDArray) -> npt.NDArray:
        """Numerically stable softmax with temperature."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted / self.temperature)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def fit(self, signals: npt.NDArray, regime_labels: npt.NDArray) -> None:
        """Train gating using softmax regression with gradient descent."""
        # Reset feature statistics for new training
        self._feat_mean = None
        self._feat_std = None
        features = self._extract_features(signals)
        n_samples, n_features = features.shape

        # One-hot encode labels
        one_hot = np.zeros((n_samples, self.n_experts))
        one_hot[np.arange(n_samples), regime_labels.astype(int)] = 1

        # Initialize weights
        self.weights = np.zeros((n_features, self.n_experts))
        self.bias = np.zeros(self.n_experts)

        # Gradient descent
        for _ in range(self.n_iterations):
            logits = features @ self.weights + self.bias
            probs = self._softmax(logits)

            # Gradient of cross-entropy loss
            grad_logits = (probs - one_hot) / n_samples
            grad_w = features.T @ grad_logits
            grad_b = grad_logits.sum(axis=0)

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

    def predict_weights(self, signals: npt.NDArray) -> npt.NDArray:
        """Predict soft weights for each expert."""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Gating network not trained. Call fit() first.")

        features = self._extract_features(signals)
        logits = features @ self.weights + self.bias
        return self._softmax(logits)


class MLPGating(GatingNetwork):
    """MLP-based gating network.

    Uses a small neural network with one or two hidden layers and ReLU activation
    for more flexible gating decisions. Trained with gradient descent.

    Parameters
    ----------
    n_experts : int
        Number of experts to route between.
    hidden_dim : int
        Number of units in the hidden layer.
    temperature : float
        Softmax temperature for output.
    n_iterations : int
        Number of training iterations.
    learning_rate : float
        Learning rate for gradient descent.
    use_two_layers : bool
        If True, use two hidden layers instead of one.
    """

    def __init__(
        self,
        n_experts: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        n_iterations: int = 3000,
        learning_rate: float = 0.01,
        use_two_layers: bool = False
    ):
        self.n_experts = n_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.use_two_layers = use_two_layers
        self.W1: npt.NDArray | None = None
        self.b1: npt.NDArray | None = None
        self.W2: npt.NDArray | None = None
        self.b2: npt.NDArray | None = None
        self.W3: npt.NDArray | None = None
        self.b3: npt.NDArray | None = None
        self._input_mean: npt.NDArray | None = None
        self._input_std: npt.NDArray | None = None

    def _relu(self, x: npt.NDArray) -> npt.NDArray:
        """ReLU activation function."""
        return np.maximum(0, x)

    def _softmax(self, logits: npt.NDArray) -> npt.NDArray:
        """Numerically stable softmax with temperature."""
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted / self.temperature)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def _preprocess(self, signals: npt.NDArray, fit: bool = False) -> npt.NDArray:
        """Preprocess signals: normalize magnitude and standardize."""
        mag = np.abs(signals)

        # Per-sample normalization (remove M0 dependency)
        mag_sum = mag.sum(axis=1, keepdims=True)
        mag_sum = np.where(mag_sum < 1e-10, 1e-10, mag_sum)
        X = mag / mag_sum

        # Standardize features
        if fit:
            self._input_mean = X.mean(axis=0)
            self._input_std = X.std(axis=0)
            self._input_std = np.where(self._input_std < 1e-10, 1.0, self._input_std)

        X = (X - self._input_mean) / self._input_std
        return X

    def fit(self, signals: npt.NDArray, regime_labels: npt.NDArray) -> None:
        """Train MLP gating network with backpropagation."""
        X = self._preprocess(signals, fit=True)
        n_samples, input_dim = X.shape

        # One-hot encode labels
        one_hot = np.zeros((n_samples, self.n_experts))
        one_hot[np.arange(n_samples), regime_labels.astype(int)] = 1

        # Xavier/He initialization
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((input_dim, self.hidden_dim)) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros(self.hidden_dim)

        if self.use_two_layers:
            hidden2_dim = self.hidden_dim // 2
            self.W2 = rng.standard_normal((self.hidden_dim, hidden2_dim)) * np.sqrt(2 / self.hidden_dim)
            self.b2 = np.zeros(hidden2_dim)
            self.W3 = rng.standard_normal((hidden2_dim, self.n_experts)) * np.sqrt(2 / hidden2_dim)
            self.b3 = np.zeros(self.n_experts)
        else:
            self.W2 = rng.standard_normal((self.hidden_dim, self.n_experts)) * np.sqrt(2 / self.hidden_dim)
            self.b2 = np.zeros(self.n_experts)

        # Training loop with learning rate decay
        for iteration in range(self.n_iterations):
            # Learning rate schedule
            lr = self.learning_rate * (0.1 ** (iteration // (self.n_iterations // 3)))

            # Forward pass
            hidden1 = self._relu(X @ self.W1 + self.b1)

            if self.use_two_layers:
                hidden2 = self._relu(hidden1 @ self.W2 + self.b2)
                logits = hidden2 @ self.W3 + self.b3
            else:
                logits = hidden1 @ self.W2 + self.b2

            probs = self._softmax(logits)

            # Backward pass
            grad_logits = (probs - one_hot) / n_samples

            if self.use_two_layers:
                grad_W3 = hidden2.T @ grad_logits
                grad_b3 = grad_logits.sum(axis=0)

                grad_hidden2 = grad_logits @ self.W3.T
                grad_hidden2[hidden2 <= 0] = 0
                grad_W2 = hidden1.T @ grad_hidden2
                grad_b2 = grad_hidden2.sum(axis=0)

                grad_hidden1 = grad_hidden2 @ self.W2.T
                grad_hidden1[hidden1 <= 0] = 0
                grad_W1 = X.T @ grad_hidden1
                grad_b1 = grad_hidden1.sum(axis=0)

                self.W3 -= lr * grad_W3
                self.b3 -= lr * grad_b3
            else:
                grad_W2 = hidden1.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)

                grad_hidden1 = grad_logits @ self.W2.T
                grad_hidden1[hidden1 <= 0] = 0
                grad_W1 = X.T @ grad_hidden1
                grad_b1 = grad_hidden1.sum(axis=0)

            # Update weights
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2

    def predict_weights(self, signals: npt.NDArray) -> npt.NDArray:
        """Predict soft weights for each expert."""
        if self.W1 is None:
            raise RuntimeError("Gating network not trained. Call fit() first.")

        X = self._preprocess(signals, fit=False)
        hidden1 = self._relu(X @ self.W1 + self.b1)

        if self.use_two_layers:
            hidden2 = self._relu(hidden1 @ self.W2 + self.b2)
            logits = hidden2 @ self.W3 + self.b3
        else:
            logits = hidden1 @ self.W2 + self.b2

        return self._softmax(logits)


class TemplateGating(GatingNetwork):
    """Template-matching gating using prototype signals.

    A physics-informed approach that computes mean signal SHAPES (normalized
    magnitudes) for each regime and uses cosine similarity to assign expert
    weights. Using magnitude avoids phase cancellation issues.

    Parameters
    ----------
    n_experts : int
        Number of experts/templates.
    temperature : float
        Softmax temperature. Lower values produce sharper matching.
    """

    def __init__(self, n_experts: int, temperature: float = 0.5):
        self.n_experts = n_experts
        self.temperature = temperature
        self.templates: npt.NDArray | None = None

    def _normalize_shape(self, signals: npt.NDArray) -> npt.NDArray:
        """Normalize to unit-sum magnitude (captures shape, not scale)."""
        mag = np.abs(signals)
        mag_sum = mag.sum(axis=1, keepdims=True)
        mag_sum = np.where(mag_sum < 1e-10, 1e-10, mag_sum)
        return mag / mag_sum

    def fit(self, signals: npt.NDArray, regime_labels: npt.NDArray) -> None:
        """Compute template (mean normalized magnitude shape) for each regime."""
        # Use normalized magnitudes to avoid phase cancellation
        normalized = self._normalize_shape(signals)

        self.templates = np.zeros(
            (self.n_experts, signals.shape[1]),
            dtype=np.float64
        )

        for k in range(self.n_experts):
            mask = regime_labels == k
            if mask.sum() > 0:
                self.templates[k] = normalized[mask].mean(axis=0)

        # Normalize templates to unit L2 norm for cosine similarity
        temp_norm = np.linalg.norm(self.templates, axis=1, keepdims=True)
        temp_norm = np.where(temp_norm < 1e-10, 1e-10, temp_norm)
        self.templates = self.templates / temp_norm

    def predict_weights(self, signals: npt.NDArray) -> npt.NDArray:
        """Predict weights based on cosine similarity to templates."""
        if self.templates is None:
            raise RuntimeError("Gating network not trained. Call fit() first.")

        # Normalize input signals the same way
        normalized = self._normalize_shape(signals)

        # L2 normalize for cosine similarity
        sig_norm = np.linalg.norm(normalized, axis=1, keepdims=True)
        sig_norm = np.where(sig_norm < 1e-10, 1e-10, sig_norm)
        signals_normalized = normalized / sig_norm

        # Cosine similarity: [n_samples, n_experts]
        similarities = signals_normalized @ self.templates.T

        # Softmax with temperature
        shifted = similarities - similarities.max(axis=1, keepdims=True)
        exp_sim = np.exp(shifted / self.temperature)
        return exp_sim / exp_sim.sum(axis=1, keepdims=True)


class TopKGating(GatingNetwork):
    """Sparse gating that only activates top-k experts.

    This reduces computation by only using the top-k experts for each input,
    similar to the sparse MoE architecture used in large language models.

    Parameters
    ----------
    n_experts : int
        Total number of experts.
    k : int
        Number of experts to activate per input.
    base_gating : GatingNetwork
        Underlying gating network to compute initial weights.
    """

    def __init__(
        self,
        n_experts: int,
        k: int = 2,
        base_gating: GatingNetwork | None = None
    ):
        self.n_experts = n_experts
        self.k = min(k, n_experts)
        self.base_gating = base_gating or MLPGating(n_experts)

    def fit(self, signals: npt.NDArray, regime_labels: npt.NDArray) -> None:
        """Train the underlying gating network."""
        self.base_gating.fit(signals, regime_labels)

    def predict_weights(self, signals: npt.NDArray) -> npt.NDArray:
        """Predict sparse weights with only top-k experts active."""
        # Get base weights
        weights = self.base_gating.predict_weights(signals)

        # Keep only top-k experts
        n_samples = weights.shape[0]
        sparse_weights = np.zeros_like(weights)

        for i in range(n_samples):
            top_k_idx = np.argsort(weights[i])[-self.k:]
            sparse_weights[i, top_k_idx] = weights[i, top_k_idx]

        # Renormalize
        row_sums = sparse_weights.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-10, 1e-10, row_sums)
        sparse_weights /= row_sums

        return sparse_weights


def create_gating_network(
    gating_type: str,
    n_experts: int,
    temperature: float = 1.0,
    **kwargs
) -> GatingNetwork:
    """Factory function for creating gating networks.

    Parameters
    ----------
    gating_type : str
        Type of gating network: 'feature', 'mlp', 'template', or 'topk'.
    n_experts : int
        Number of experts.
    temperature : float
        Softmax temperature.
    **kwargs
        Additional arguments passed to the gating network constructor.

    Returns
    -------
    GatingNetwork
        Initialized gating network.
    """
    if gating_type == 'feature':
        return FeatureGating(n_experts, temperature, **kwargs)
    elif gating_type == 'mlp':
        return MLPGating(n_experts, temperature=temperature, **kwargs)
    elif gating_type == 'template':
        return TemplateGating(n_experts, temperature)
    elif gating_type == 'topk':
        k = kwargs.pop('k', 2)
        base_type = kwargs.pop('base_gating_type', 'mlp')
        base_gating = create_gating_network(base_type, n_experts, temperature, **kwargs)
        return TopKGating(n_experts, k, base_gating)
    else:
        raise ValueError(f"Unknown gating type: {gating_type}")
