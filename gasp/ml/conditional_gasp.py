"""
Conditional GASP Model: Neural network that predicts tissue-specific GASP coefficients.

The model learns to infer latent tissue properties from bSSFP signals and predicts
polynomial coefficients that adapt to different T2/T1 ratios.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from pathlib import Path
from itertools import combinations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for Conditional GASP. "
            "Install with: pip install torch"
        )


def design_matrix_torch(X: "torch.Tensor", method: str) -> "torch.Tensor":
    """
    Build the design matrix Phi for the chosen method (PyTorch version).

    Methods:
      - 'linear'      : Phi = [X]
      - 'affine'      : Phi = [1, X]
      - 'quad'        : Phi = [1, X, X^2]
      - 'quad-cross'  : Phi = [1, X, X^2, {X_i X_j}_{i<j}]
    """
    _check_torch()
    method = method.lower()
    n = X.shape[0]

    # Ensure consistent dtype (complex64 for complex, float32 for real)
    if torch.is_complex(X):
        X = X.to(torch.complex64)
        ones = torch.ones((n, 1), dtype=torch.complex64, device=X.device)
    else:
        X = X.float()
        ones = torch.ones((n, 1), dtype=torch.float32, device=X.device)

    if method == "linear":
        return X

    if method == "affine":
        return torch.cat([ones, X], dim=-1)

    if method == "quad":
        return torch.cat([ones, X, X**2], dim=-1)

    if method == "quad-cross":
        p = X.shape[1]
        crosses = [X[:, i:i+1] * X[:, j:j+1] for i, j in combinations(range(p), 2)]
        if crosses:
            cross_block = torch.cat(crosses, dim=-1)
        else:
            cross_block = torch.empty((n, 0), dtype=X.dtype, device=X.device)
        return torch.cat([ones, X, X**2, cross_block], dim=-1)

    raise ValueError(f"Unknown method '{method}'. Choose from "
                     f"{{'linear','affine','quad','quad-cross'}}.")


def get_n_coeffs(n_acquisitions: int, method: str) -> int:
    """Calculate number of coefficients for given method and acquisition count."""
    method = method.lower()
    if method == "linear":
        return n_acquisitions
    if method == "affine":
        return n_acquisitions + 1
    if method == "quad":
        return 2 * n_acquisitions + 1
    if method == "quad-cross":
        p = n_acquisitions
        n_crosses = p * (p - 1) // 2
        return 2 * p + 1 + n_crosses
    raise ValueError(f"Unknown method '{method}'")


class LearnedGASP(nn.Module):
    """
    Learned GASP: Optimize global coefficients via gradient descent over diverse training data.

    Unlike standard GASP which fits coefficients on a single T1/T2 using least squares,
    LearnedGASP learns coefficients that work well across a range of tissue types.

    This serves as an ablation: comparing LearnedGASP vs ConditionalGASP shows the
    benefit of per-sample adaptive coefficients vs a single global set.

    Args:
        n_acquisitions: Number of acquisition points (phase-cycles x TRs)
        method: GASP method ('linear', 'affine', 'quad', 'quad-cross')
    """

    def __init__(
        self,
        n_acquisitions: int,
        method: str = "affine",
    ):
        _check_torch()
        super().__init__()

        self.n_acquisitions = n_acquisitions
        self.method = method
        self.n_coeffs = get_n_coeffs(n_acquisitions, method)

        # Learnable coefficients (real and imaginary parts)
        self.A_real = nn.Parameter(torch.randn(self.n_coeffs) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(self.n_coeffs) * 0.01)

    @property
    def coefficients(self) -> "torch.Tensor":
        """Get complex coefficients."""
        return torch.complex(self.A_real, self.A_imag)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Apply GASP with learned coefficients.

        Args:
            x: Complex signal tensor [batch, n_acquisitions]

        Returns:
            output: GASP output [batch]
        """
        # Build design matrix
        if torch.is_complex(x):
            Phi = design_matrix_torch(x, self.method)
        else:
            x_complex = torch.complex(
                x[..., :self.n_acquisitions],
                x[..., self.n_acquisitions:]
            )
            Phi = design_matrix_torch(x_complex, self.method)

        # Apply coefficients: output = Phi @ A
        A = self.coefficients
        output = (Phi * A).sum(dim=-1)
        return output

    def get_numpy_coefficients(self) -> npt.NDArray:
        """Get coefficients as numpy array for use with standard GASP."""
        with torch.no_grad():
            return self.coefficients.cpu().numpy()

    def save(self, path: str | Path):
        """Save model weights and config."""
        _check_torch()
        path = Path(path)
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'n_acquisitions': self.n_acquisitions,
                'method': self.method,
            }
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu') -> "LearnedGASP":
        """Load model from saved checkpoint."""
        _check_torch()
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint['config'])
        model = model.to(device)  # Move model to device before loading state dict
        model.load_state_dict(checkpoint['state_dict'])
        return model


class ConditionalGASP(nn.Module):
    """
    Conditional GASP: Neural network that predicts tissue-specific polynomial coefficients.

    Architecture:
        Input Signal: x in C^(n_acquisitions)
            |
        Tissue Encoder: z = E(x) in R^d  (latent tissue embedding)
            |
        Coefficient Predictor: A = C(z) in C^k  (GASP polynomial coefficients)
            |
        Output: y = Phi(x) @ A  (uses standard GASP design matrix)

    Args:
        n_acquisitions: Number of acquisition points (phase-cycles x TRs)
        latent_dim: Dimension of latent tissue embedding
        method: GASP method ('linear', 'affine', 'quad', 'quad-cross')
        hidden_dims: Hidden layer dimensions for encoder
    """

    def __init__(
        self,
        n_acquisitions: int,
        latent_dim: int = 16,
        method: str = "affine",
        hidden_dims: tuple[int, ...] = (128, 64),
    ):
        _check_torch()
        super().__init__()

        self.n_acquisitions = n_acquisitions
        self.latent_dim = latent_dim
        self.method = method
        self.n_coeffs = get_n_coeffs(n_acquisitions, method)

        # Encoder: signal -> latent tissue embedding
        # Input is real/imag concatenated, so 2 * n_acquisitions
        encoder_layers = []
        in_dim = n_acquisitions * 2
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Coefficient predictor: latent -> GASP coefficients
        # Output is real/imag for complex coefficients
        self.coeff_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], self.n_coeffs * 2),
        )

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Encode signal to latent tissue embedding.

        Args:
            x: Complex signal tensor [batch, n_acquisitions] or [batch, n_acquisitions] real/imag

        Returns:
            z: Latent tissue embedding [batch, latent_dim]
        """
        if torch.is_complex(x):
            x_real_imag = torch.cat([x.real, x.imag], dim=-1)
        else:
            x_real_imag = x
        # Ensure float32 dtype for compatibility with model weights
        x_real_imag = x_real_imag.float()
        return self.encoder(x_real_imag)

    def predict_coefficients(self, z: "torch.Tensor") -> "torch.Tensor":
        """
        Predict GASP coefficients from latent embedding.

        Args:
            z: Latent tissue embedding [batch, latent_dim]

        Returns:
            A: Complex coefficients [batch, n_coeffs]
        """
        A_flat = self.coeff_predictor(z)
        A_real = A_flat[..., :self.n_coeffs]
        A_imag = A_flat[..., self.n_coeffs:]
        return torch.complex(A_real, A_imag)

    def forward(
        self,
        x: "torch.Tensor",
        return_latent: bool = False
    ) -> tuple["torch.Tensor", "torch.Tensor"] | "torch.Tensor":
        """
        Forward pass: encode signal and predict coefficients.

        Args:
            x: Complex signal tensor [batch, n_acquisitions]
            return_latent: If True, also return latent embedding

        Returns:
            A: Complex coefficients [batch, n_coeffs]
            z: (optional) Latent embedding [batch, latent_dim]
        """
        z = self.encode(x)
        A = self.predict_coefficients(z)

        if return_latent:
            return A, z
        return A

    def apply_gasp(self, x: "torch.Tensor", A: "torch.Tensor" = None) -> "torch.Tensor":
        """
        Apply GASP with predicted or provided coefficients.

        Args:
            x: Complex signal tensor [batch, n_acquisitions]
            A: Optional coefficients. If None, predicts from x.

        Returns:
            output: GASP output [batch]
        """
        if A is None:
            A = self.forward(x)

        # Build design matrix
        if torch.is_complex(x):
            Phi = design_matrix_torch(x, self.method)
        else:
            x_complex = torch.complex(
                x[..., :self.n_acquisitions],
                x[..., self.n_acquisitions:]
            )
            Phi = design_matrix_torch(x_complex, self.method)

        # Apply coefficients: output = Phi @ A
        # Phi: [batch, n_coeffs], A: [batch, n_coeffs]
        output = (Phi * A).sum(dim=-1)
        return output

    def save(self, path: str | Path):
        """Save model weights and config."""
        _check_torch()
        path = Path(path)
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'n_acquisitions': self.n_acquisitions,
                'latent_dim': self.latent_dim,
                'method': self.method,
            }
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = 'cpu') -> "ConditionalGASP":
        """Load model from saved checkpoint."""
        _check_torch()
        path = Path(path)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls(**checkpoint['config'])
        model = model.to(device)  # Move model to device before loading state dict
        model.load_state_dict(checkpoint['state_dict'])
        return model


def run_conditional_gasp(
    I: npt.NDArray,
    model: ConditionalGASP,
    device: str = 'cpu'
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Apply trained Conditional GASP to image data.

    Args:
        I: Input data [H, W, n_acquisitions] complex
        model: Trained ConditionalGASP model
        device: PyTorch device

    Returns:
        output: GASP output image [H, W]
        coefficients: Predicted coefficients [H, W, n_coeffs]
        latent: Latent embeddings [H, W, latent_dim]
    """
    _check_torch()

    h, w = I.shape[:2]
    X = I.reshape(h * w, -1)

    # Convert to torch
    X_torch = torch.from_numpy(X).to(device)

    model.eval()
    with torch.no_grad():
        A, z = model(X_torch, return_latent=True)
        output = model.apply_gasp(X_torch, A)

    # Convert back to numpy
    output_np = output.cpu().numpy().reshape(h, w)
    A_np = A.cpu().numpy().reshape(h, w, -1)
    z_np = z.cpu().numpy().reshape(h, w, -1)

    return output_np, A_np, z_np
