"""
Loss Functions for Conditional GASP Training.

Provides various loss functions for training the Conditional GASP model,
including spectral profile reconstruction, coefficient regularization,
and latent space smoothness losses.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for loss functions.")


def spectral_profile_loss(
    pred_output: "torch.Tensor",
    target_profile: "torch.Tensor",
    reduction: str = 'mean'
) -> "torch.Tensor":
    """
    Compute MSE loss between predicted GASP output and target profile.

    Args:
        pred_output: Predicted output [batch, width] (complex)
        target_profile: Target profile [batch, width] or [width] (real)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value
    """
    _check_torch()

    # Use magnitude of complex output
    pred_mag = pred_output.abs()

    # Expand target if needed
    if target_profile.dim() == 1:
        target_profile = target_profile.unsqueeze(0).expand(pred_mag.shape[0], -1)

    return F.mse_loss(pred_mag, target_profile, reduction=reduction)


def coefficient_l2_loss(
    coefficients: "torch.Tensor",
    reduction: str = 'mean'
) -> "torch.Tensor":
    """
    L2 regularization on predicted coefficients to prevent explosion.

    Args:
        coefficients: Predicted coefficients [batch, n_coeffs] (complex)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value
    """
    _check_torch()

    coeff_mag_sq = coefficients.abs() ** 2

    if reduction == 'mean':
        return coeff_mag_sq.mean()
    elif reduction == 'sum':
        return coeff_mag_sq.sum()
    return coeff_mag_sq


def embedding_smoothness_loss(
    z: "torch.Tensor",
    tissue_params: "torch.Tensor" = None,
    temperature: float = 1.0,
) -> "torch.Tensor":
    """
    Encourage similar T2/T1 ratios to have similar embeddings.

    If tissue_params is provided, uses supervised contrastive-style loss.
    Otherwise, uses self-supervised smoothness in embedding space.

    Args:
        z: Latent embeddings [batch, latent_dim]
        tissue_params: Optional tissue parameters [batch, 3] (T1, T2, ratio)
        temperature: Temperature for similarity scaling

    Returns:
        loss: Scalar loss value
    """
    _check_torch()

    if tissue_params is None:
        # Self-supervised: encourage smooth embedding manifold
        # Penalize large variations between adjacent samples
        z_norm = F.normalize(z, dim=-1)
        similarity = z_norm @ z_norm.T
        # Encourage high similarity on average (smooth manifold)
        return -similarity.mean()

    # Supervised: similar T2/T1 ratios should have similar embeddings
    ratios = tissue_params[:, 2]  # T2/T1 ratio

    # Compute pairwise ratio differences
    ratio_diff = (ratios.unsqueeze(1) - ratios.unsqueeze(0)).abs()

    # Compute pairwise embedding similarities
    z_norm = F.normalize(z, dim=-1)
    similarity = z_norm @ z_norm.T

    # Loss: high similarity when ratio_diff is small, low when large
    # Use soft weighting based on ratio difference
    weights = torch.exp(-ratio_diff / temperature)

    # Weighted MSE: embeddings should be similar when ratios are similar
    loss = ((1 - similarity) * weights).mean()

    return loss


def contrastive_tissue_loss(
    z: "torch.Tensor",
    tissue_params: "torch.Tensor",
    temperature: float = 0.1,
    ratio_threshold: float = 0.05,
) -> "torch.Tensor":
    """
    Contrastive loss that pulls together embeddings with similar T2/T1 ratios.

    Args:
        z: Latent embeddings [batch, latent_dim]
        tissue_params: Tissue parameters [batch, 3] (T1, T2, ratio)
        temperature: Temperature for softmax
        ratio_threshold: Ratio difference threshold for positive pairs

    Returns:
        loss: Scalar loss value
    """
    _check_torch()

    batch_size = z.shape[0]
    ratios = tissue_params[:, 2]  # T2/T1 ratio

    # Normalize embeddings
    z_norm = F.normalize(z, dim=-1)

    # Compute similarity matrix
    similarity = z_norm @ z_norm.T / temperature

    # Create positive mask: pairs with similar T2/T1 ratios
    ratio_diff = (ratios.unsqueeze(1) - ratios.unsqueeze(0)).abs()
    positive_mask = (ratio_diff < ratio_threshold).float()
    positive_mask.fill_diagonal_(0)  # Exclude self-similarity

    # InfoNCE-style loss
    # For each sample, pull together positives, push apart negatives
    exp_sim = torch.exp(similarity)
    exp_sim.fill_diagonal_(0)  # Exclude self

    # Sum of similarities with positive samples
    pos_sum = (exp_sim * positive_mask).sum(dim=1)

    # Sum of all similarities (denominator)
    all_sum = exp_sim.sum(dim=1)

    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    loss = -torch.log((pos_sum + eps) / (all_sum + eps))

    # Only compute loss for samples that have positives
    has_positives = positive_mask.sum(dim=1) > 0
    if has_positives.sum() > 0:
        return loss[has_positives].mean()

    return torch.tensor(0.0, device=z.device)


def conditional_gasp_loss(
    pred_output: "torch.Tensor",
    target_profile: "torch.Tensor",
    coefficients: "torch.Tensor",
    z: "torch.Tensor",
    tissue_params: "torch.Tensor" = None,
    lambda_l2: float = 1e-3,
    lambda_smooth: float = 1e-4,
    lambda_contrastive: float = 0.0,
) -> dict["str", "torch.Tensor"]:
    """
    Combined loss function for Conditional GASP training.

    Args:
        pred_output: Predicted GASP output [batch, width] (complex)
        target_profile: Target profile [batch, width] or [width] (real)
        coefficients: Predicted coefficients [batch, n_coeffs] (complex)
        z: Latent embeddings [batch, latent_dim]
        tissue_params: Optional tissue parameters [batch, 3]
        lambda_l2: Weight for coefficient L2 regularization
        lambda_smooth: Weight for embedding smoothness loss
        lambda_contrastive: Weight for contrastive tissue loss

    Returns:
        dict with 'total', 'profile', 'l2', 'smooth', 'contrastive' losses
    """
    _check_torch()

    # Primary loss: spectral profile reconstruction
    L_profile = spectral_profile_loss(pred_output, target_profile)

    # Regularization: prevent coefficient explosion
    L_l2 = coefficient_l2_loss(coefficients)

    # Smoothness: encourage similar tissues to have similar embeddings
    L_smooth = embedding_smoothness_loss(z, tissue_params)

    # Optional contrastive loss
    if lambda_contrastive > 0 and tissue_params is not None:
        L_contrastive = contrastive_tissue_loss(z, tissue_params)
    else:
        L_contrastive = torch.tensor(0.0, device=pred_output.device)

    # Total loss
    total = (
        L_profile +
        lambda_l2 * L_l2 +
        lambda_smooth * L_smooth +
        lambda_contrastive * L_contrastive
    )

    return {
        'total': total,
        'profile': L_profile,
        'l2': L_l2,
        'smooth': L_smooth,
        'contrastive': L_contrastive,
    }


class ConditionalGASPLoss(nn.Module):
    """
    PyTorch Module wrapper for Conditional GASP loss.

    Makes it easy to use with standard training loops.
    """

    def __init__(
        self,
        lambda_l2: float = 1e-3,
        lambda_smooth: float = 1e-4,
        lambda_contrastive: float = 0.0,
    ):
        _check_torch()
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_smooth = lambda_smooth
        self.lambda_contrastive = lambda_contrastive

    def forward(
        self,
        pred_output: "torch.Tensor",
        target_profile: "torch.Tensor",
        coefficients: "torch.Tensor",
        z: "torch.Tensor",
        tissue_params: "torch.Tensor" = None,
    ) -> dict["str", "torch.Tensor"]:
        return conditional_gasp_loss(
            pred_output,
            target_profile,
            coefficients,
            z,
            tissue_params,
            self.lambda_l2,
            self.lambda_smooth,
            self.lambda_contrastive,
        )
