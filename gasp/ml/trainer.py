"""
Trainer for Conditional GASP Model.

Provides training loop, validation, and utilities for training
the Conditional GASP neural network.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Callable

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .conditional_gasp import ConditionalGASP, LearnedGASP, design_matrix_torch
from .data_generator import generate_training_batch, ConditionalGASPDataset
from .losses import conditional_gasp_loss, ConditionalGASPLoss
from gasp.simulation import SSFPParams


def _check_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")


class TrainingHistory:
    """Track training metrics over epochs."""

    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.profile_loss = []
        self.l2_loss = []
        self.smooth_loss = []

    def update(self, epoch: int, metrics: dict):
        self.epochs.append(epoch)
        self.train_loss.append(metrics.get('train_loss', 0))
        self.val_loss.append(metrics.get('val_loss', 0))
        self.profile_loss.append(metrics.get('profile_loss', 0))
        self.l2_loss.append(metrics.get('l2_loss', 0))
        self.smooth_loss.append(metrics.get('smooth_loss', 0))

    def to_dict(self) -> dict:
        return {
            'epochs': self.epochs,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'profile_loss': self.profile_loss,
            'l2_loss': self.l2_loss,
            'smooth_loss': self.smooth_loss,
        }


def train_epoch(
    model: ConditionalGASP,
    optimizer: "optim.Optimizer",
    params: SSFPParams,
    target_profile: "torch.Tensor",
    batch_size: int,
    n_batches: int,
    loss_fn: ConditionalGASPLoss,
    device: str,
    t1_range: tuple[float, float],
    t2_t1_ratio_range: tuple[float, float],
    width: int,
    noise_sigma: float,
) -> dict:
    """
    Train for one epoch with on-the-fly data generation.

    Returns:
        dict with average losses for the epoch
    """
    _check_torch()
    model.train()

    total_loss = 0
    total_profile = 0
    total_l2 = 0
    total_smooth = 0

    iterator = range(n_batches)
    if TQDM_AVAILABLE:
        iterator = tqdm(iterator, desc="Training", leave=False)

    for _ in iterator:
        # Generate fresh batch
        signals, tissue_params = generate_training_batch(
            batch_size, params, t1_range, t2_t1_ratio_range,
            width=width, noise_sigma=noise_sigma
        )

        # Convert to torch tensors
        signals = torch.from_numpy(signals).to(device)
        tissue_params_t = torch.from_numpy(tissue_params).float().to(device)

        # Flatten spatial dimension: [batch, width, n_acq] -> [batch*width, n_acq]
        batch, w, n_acq = signals.shape
        signals_flat = signals.reshape(batch * w, n_acq)

        # Forward pass
        A, z = model(signals_flat, return_latent=True)

        # Build design matrix and compute output
        Phi = design_matrix_torch(signals_flat, model.method)
        output = (Phi * A).sum(dim=-1)

        # Tile target profile for batch
        target_tiled = target_profile.unsqueeze(0).expand(batch, -1).reshape(-1)

        # Tile tissue params for each frequency point
        tissue_params_tiled = tissue_params_t.unsqueeze(1).expand(-1, w, -1).reshape(-1, 3)

        # Compute loss
        losses = loss_fn(output, target_tiled, A, z, tissue_params_tiled)

        # Backprop
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # Accumulate
        total_loss += losses['total'].item()
        total_profile += losses['profile'].item()
        total_l2 += losses['l2'].item()
        total_smooth += losses['smooth'].item()

    return {
        'train_loss': total_loss / n_batches,
        'profile_loss': total_profile / n_batches,
        'l2_loss': total_l2 / n_batches,
        'smooth_loss': total_smooth / n_batches,
    }


def validate(
    model: ConditionalGASP,
    params: SSFPParams,
    target_profile: "torch.Tensor",
    loss_fn: ConditionalGASPLoss,
    device: str,
    n_samples: int = 1000,
    t1_range: tuple[float, float] = (0.1, 4.0),
    t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
    width: int = 256,
) -> float:
    """Validate on held-out data."""
    _check_torch()
    model.eval()

    with torch.no_grad():
        signals, tissue_params = generate_training_batch(
            n_samples, params, t1_range, t2_t1_ratio_range, width=width
        )

        signals = torch.from_numpy(signals).to(device)
        tissue_params_t = torch.from_numpy(tissue_params).float().to(device)

        batch, w, n_acq = signals.shape
        signals_flat = signals.reshape(batch * w, n_acq)

        A, z = model(signals_flat, return_latent=True)
        Phi = design_matrix_torch(signals_flat, model.method)
        output = (Phi * A).sum(dim=-1)

        target_tiled = target_profile.unsqueeze(0).expand(batch, -1).reshape(-1)
        tissue_params_tiled = tissue_params_t.unsqueeze(1).expand(-1, w, -1).reshape(-1, 3)

        losses = loss_fn(output, target_tiled, A, z, tissue_params_tiled)

    return losses['total'].item()


def train_conditional_gasp(
    params: SSFPParams,
    target_profile: npt.NDArray,
    n_epochs: int = 100,
    batch_size: int = 32,
    n_batches_per_epoch: int = 50,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    latent_dim: int = 16,
    method: str = "affine",
    t1_range: tuple[float, float] = (0.1, 4.0),
    t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
    width: int = 256,
    noise_sigma: float = 0.005,
    lambda_l2: float = 1e-3,
    lambda_smooth: float = 1e-4,
    lambda_contrastive: float = 0.0,
    device: str = None,
    save_path: str | Path = None,
    save_best: bool = True,
    verbose: bool = True,
) -> tuple[ConditionalGASP, TrainingHistory]:
    """
    Train a Conditional GASP model.

    Args:
        params: SSFPParams object with acquisition parameters
        target_profile: Target spectral profile [width]
        n_epochs: Number of training epochs
        batch_size: Batch size (number of T1/T2 samples per batch)
        n_batches_per_epoch: Number of batches per epoch
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        latent_dim: Dimension of latent tissue embedding
        method: GASP method ('linear', 'affine', 'quad', 'quad-cross')
        t1_range: (min, max) T1 values in seconds
        t2_t1_ratio_range: (min, max) T2/T1 ratio values
        width: Number of frequency points
        noise_sigma: Standard deviation of training noise
        lambda_l2: Weight for coefficient L2 regularization
        lambda_smooth: Weight for embedding smoothness loss
        lambda_contrastive: Weight for contrastive tissue loss
        device: PyTorch device ('cuda', 'cpu', or None for auto)
        save_path: Path to save best model
        save_best: If True, save best model during training
        verbose: Print training progress

    Returns:
        model: Trained ConditionalGASP model
        history: TrainingHistory with loss curves
    """
    _check_torch()

    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Training on device: {device}")

    # Get number of acquisitions from params
    n_acquisitions = params.length

    # Initialize model
    model = ConditionalGASP(
        n_acquisitions=n_acquisitions,
        latent_dim=latent_dim,
        method=method,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    # Convert target profile to tensor
    target_profile_t = torch.from_numpy(target_profile).float().to(device)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    # Loss function
    loss_fn = ConditionalGASPLoss(
        lambda_l2=lambda_l2,
        lambda_smooth=lambda_smooth,
        lambda_contrastive=lambda_contrastive,
    )

    # Training history
    history = TrainingHistory()
    best_val_loss = float('inf')

    # Training loop
    epoch_iterator = range(n_epochs)
    if verbose and TQDM_AVAILABLE:
        epoch_iterator = tqdm(epoch_iterator, desc="Epochs")

    for epoch in epoch_iterator:
        # Train
        train_metrics = train_epoch(
            model, optimizer, params, target_profile_t,
            batch_size, n_batches_per_epoch, loss_fn, device,
            t1_range, t2_t1_ratio_range, width, noise_sigma
        )

        # Validate
        val_loss = validate(
            model, params, target_profile_t, loss_fn, device,
            n_samples=100, t1_range=t1_range,
            t2_t1_ratio_range=t2_t1_ratio_range, width=width
        )

        # Update scheduler
        scheduler.step()

        # Record history
        metrics = {**train_metrics, 'val_loss': val_loss}
        history.update(epoch, metrics)

        # Save best model
        if save_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                model.save(save_path)

        # Log progress
        if verbose and not TQDM_AVAILABLE:
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: train_loss={train_metrics['train_loss']:.4f}, "
                      f"val_loss={val_loss:.4f}")

    if verbose:
        print(f"Training complete. Best val_loss: {best_val_loss:.4f}")

    # Load best model if saved
    if save_best and save_path and Path(save_path).exists():
        model = ConditionalGASP.load(save_path, device=device)

    return model, history


def train_learned_gasp(
    params: SSFPParams,
    target_profile: npt.NDArray,
    n_epochs: int = 100,
    batch_size: int = 32,
    n_batches_per_epoch: int = 50,
    learning_rate: float = 1e-2,
    method: str = "affine",
    t1_range: tuple[float, float] = (0.1, 4.0),
    t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
    width: int = 256,
    noise_sigma: float = 0.005,
    device: str = None,
    verbose: bool = True,
) -> tuple[LearnedGASP, TrainingHistory]:
    """
    Train a LearnedGASP model (global coefficients via gradient descent).

    Unlike standard GASP which fits on a single T1/T2, this learns coefficients
    that work well across a range of tissue types.

    Args:
        params: SSFPParams object with acquisition parameters
        target_profile: Target spectral profile [width]
        n_epochs: Number of training epochs
        batch_size: Batch size (number of T1/T2 samples per batch)
        n_batches_per_epoch: Number of batches per epoch
        learning_rate: Learning rate (typically higher than ConditionalGASP)
        method: GASP method ('linear', 'affine', 'quad', 'quad-cross')
        t1_range: (min, max) T1 values in seconds
        t2_t1_ratio_range: (min, max) T2/T1 ratio values
        width: Number of frequency points
        noise_sigma: Standard deviation of training noise
        device: PyTorch device ('cuda', 'cpu', or None for auto)
        verbose: Print training progress

    Returns:
        model: Trained LearnedGASP model
        history: TrainingHistory with loss curves
    """
    _check_torch()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"Training LearnedGASP on device: {device}")

    n_acquisitions = params.length

    model = LearnedGASP(
        n_acquisitions=n_acquisitions,
        method=method,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,} (just coefficients)")

    target_profile_t = torch.from_numpy(target_profile).float().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = TrainingHistory()

    epoch_iterator = range(n_epochs)
    if verbose and TQDM_AVAILABLE:
        epoch_iterator = tqdm(epoch_iterator, desc="Training LearnedGASP")

    for epoch in epoch_iterator:
        model.train()
        total_loss = 0

        for _ in range(n_batches_per_epoch):
            signals, _ = generate_training_batch(
                batch_size, params, t1_range, t2_t1_ratio_range,
                width=width, noise_sigma=noise_sigma
            )

            signals = torch.from_numpy(signals).to(device)
            batch, w, n_acq = signals.shape
            signals_flat = signals.reshape(batch * w, n_acq)

            output = model(signals_flat)
            target_tiled = target_profile_t.unsqueeze(0).expand(batch, -1).reshape(-1)

            loss = torch.mean((output.abs() - target_tiled) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / n_batches_per_epoch
        history.update(epoch, {'train_loss': avg_loss, 'val_loss': avg_loss,
                               'profile_loss': avg_loss, 'l2_loss': 0, 'smooth_loss': 0})

        if verbose and not TQDM_AVAILABLE and epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.6f}")

    if verbose:
        print(f"Training complete. Final loss: {avg_loss:.6f}")

    return model, history


def evaluate_model(
    model: ConditionalGASP,
    params: SSFPParams,
    target_profile: npt.NDArray,
    tissue_params_list: list[tuple[float, float]],
    width: int = 256,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate model on specific tissue types.

    Args:
        model: Trained ConditionalGASP model
        params: SSFPParams object
        target_profile: Target spectral profile
        tissue_params_list: List of (T1, T2) tuples to evaluate
        width: Number of frequency points
        device: PyTorch device

    Returns:
        dict with per-tissue evaluation metrics
    """
    _check_torch()

    from gasp.ml.data_generator import generate_signal_simple

    model.eval()
    results = {}

    target_t = torch.from_numpy(target_profile).float().to(device)

    with torch.no_grad():
        for T1, T2 in tissue_params_list:
            # Generate signal
            signal = generate_signal_simple(T1, T2, params, width=width)
            signal_t = torch.from_numpy(signal).to(device)

            # Forward pass
            A, z = model(signal_t, return_latent=True)
            Phi = design_matrix_torch(signal_t, model.method)
            output = (Phi * A).sum(dim=-1)

            # Compute error
            error = (output.abs() - target_t).pow(2).mean().item()
            mae = (output.abs() - target_t).abs().mean().item()

            results[(T1, T2)] = {
                'mse': error,
                'mae': mae,
                'output': output.cpu().numpy(),
                'latent': z.cpu().numpy(),
                'coefficients': A.cpu().numpy(),
            }

    return results
