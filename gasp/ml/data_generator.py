"""
Training Data Generator for Conditional GASP.

Generates bSSFP signals spanning the physiological T1/T2 parameter space
for training the Conditional GASP model.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Callable

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import GASP simulation functions
from gasp.ssfp import ssfp
from gasp.simulation import SSFPParams


# Physiological T1/T2 ranges (in seconds) at 1.5T/3T
# Reference values for common tissues
TISSUE_PARAMS = {
    'water': {'t1': 4.0, 't2': 2.0, 't2_t1_ratio': 0.5},
    'csf': {'t1': 4.0, 't2': 2.0, 't2_t1_ratio': 0.5},
    'gray_matter': {'t1': 0.9, 't2': 0.1, 't2_t1_ratio': 0.11},
    'white_matter': {'t1': 0.6, 't2': 0.08, 't2_t1_ratio': 0.13},
    'muscle': {'t1': 0.9, 't2': 0.05, 't2_t1_ratio': 0.056},
    'fat': {'t1': 0.25, 't2': 0.07, 't2_t1_ratio': 0.28},
    'liver': {'t1': 0.5, 't2': 0.04, 't2_t1_ratio': 0.08},
    'cartilage': {'t1': 1.0, 't2': 0.04, 't2_t1_ratio': 0.04},
    'tendon': {'t1': 0.4, 't2': 0.005, 't2_t1_ratio': 0.0125},
}


def generate_signal_simple(
    T1: float,
    T2: float,
    params: SSFPParams,
    width: int = 256,
    gradient: float = 2 * np.pi,
    min_TR: float = 5e-3,
    f0: float = 0.0,
) -> npt.NDArray:
    """
    Generate a single bSSFP signal for given T1/T2 values.

    Args:
        T1: Longitudinal relaxation time (seconds)
        T2: Transverse relaxation time (seconds)
        params: SSFPParams object with acquisition parameters
        width: Number of frequency points
        gradient: Maximum phase accumulation
        min_TR: Minimum TR for frequency calculation
        f0: Off-resonance frequency offset (Hz)

    Returns:
        signal: Complex signal [width, n_acquisitions]
    """
    # Create frequency axis
    beta = np.linspace(-gradient, gradient, width)
    f = beta / min_TR / (2 * np.pi)
    f = f + f0

    # Generate signal for each acquisition
    n_acq = params.length
    signal = np.empty((width, n_acq), dtype=np.complex128)

    for ii in range(n_acq):
        alpha = params.getAlpha(ii)
        pc = params.getPC(ii)
        TR = params.getTR(ii)
        TE = TR / 2.0
        signal[:, ii] = ssfp(T1, T2, TR, TE, alpha, pc, field_map=f)

    return signal


def generate_training_batch(
    batch_size: int,
    params: SSFPParams,
    t1_range: tuple[float, float] = (0.1, 4.0),
    t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
    width: int = 256,
    gradient: float = 2 * np.pi,
    min_TR: float = 5e-3,
    f0_range: tuple[float, float] = (0.0, 0.0),
    noise_sigma: float = 0.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Generate a batch of training signals spanning T1/T2 parameter space.

    Args:
        batch_size: Number of signals to generate
        params: SSFPParams object with acquisition parameters
        t1_range: (min, max) T1 values in seconds
        t2_t1_ratio_range: (min, max) T2/T1 ratio values
        width: Number of frequency points per signal
        gradient: Maximum phase accumulation
        min_TR: Minimum TR for frequency calculation
        f0_range: (min, max) off-resonance frequency offset (Hz)
        noise_sigma: Standard deviation of Gaussian noise (0 = no noise)

    Returns:
        signals: Complex signals [batch_size, width, n_acquisitions]
        tissue_params: Tissue parameters [batch_size, 3] (T1, T2, T2/T1 ratio)
    """
    n_acq = params.length
    signals = np.empty((batch_size, width, n_acq), dtype=np.complex128)
    tissue_params = np.empty((batch_size, 3))

    for i in range(batch_size):
        # Sample T1 uniformly
        T1 = np.random.uniform(*t1_range)

        # Sample T2/T1 ratio uniformly, then compute T2
        ratio = np.random.uniform(*t2_t1_ratio_range)
        T2 = T1 * ratio

        # Optional off-resonance variation
        f0 = np.random.uniform(*f0_range)

        # Generate signal
        signal = generate_signal_simple(
            T1, T2, params, width, gradient, min_TR, f0
        )

        # Add noise if specified
        if noise_sigma > 0:
            noise = noise_sigma * (
                np.random.randn(*signal.shape) +
                1j * np.random.randn(*signal.shape)
            )
            signal = signal + noise

        signals[i] = signal
        tissue_params[i] = [T1, T2, ratio]

    return signals, tissue_params


def generate_training_dataset(
    n_samples: int,
    params: SSFPParams,
    target_profile_fn: Callable[[int], npt.NDArray],
    t1_range: tuple[float, float] = (0.1, 4.0),
    t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
    width: int = 256,
    gradient: float = 2 * np.pi,
    min_TR: float = 5e-3,
    noise_sigma: float = 0.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Generate a complete training dataset with target profiles.

    Args:
        n_samples: Total number of samples
        params: SSFPParams object with acquisition parameters
        target_profile_fn: Function that generates target profile given width
        t1_range: (min, max) T1 values in seconds
        t2_t1_ratio_range: (min, max) T2/T1 ratio values
        width: Number of frequency points
        gradient: Maximum phase accumulation
        min_TR: Minimum TR for frequency calculation
        noise_sigma: Standard deviation of Gaussian noise

    Returns:
        signals: Complex signals [n_samples, width, n_acquisitions]
        tissue_params: Tissue parameters [n_samples, 3]
        targets: Target profiles [n_samples, width]
    """
    signals, tissue_params = generate_training_batch(
        n_samples, params, t1_range, t2_t1_ratio_range,
        width, gradient, min_TR, noise_sigma=noise_sigma
    )

    # Generate target profile (same for all samples)
    target = target_profile_fn(width)
    targets = np.tile(target, (n_samples, 1))

    return signals, tissue_params, targets


def sample_tissue_specific(
    batch_size: int,
    params: SSFPParams,
    tissue_names: list[str] | None = None,
    noise_sigma: float = 0.0,
    width: int = 256,
    gradient: float = 2 * np.pi,
    min_TR: float = 5e-3,
    t1_t2_jitter: float = 0.1,
) -> tuple[npt.NDArray, npt.NDArray, list[str]]:
    """
    Generate signals from specific tissue types with optional parameter jitter.

    Args:
        batch_size: Number of signals to generate
        params: SSFPParams object
        tissue_names: List of tissue names to sample from (None = all)
        noise_sigma: Standard deviation of Gaussian noise
        width: Number of frequency points
        gradient: Maximum phase accumulation
        min_TR: Minimum TR for frequency calculation
        t1_t2_jitter: Relative jitter to add to T1/T2 values (0.1 = 10%)

    Returns:
        signals: Complex signals [batch_size, width, n_acquisitions]
        tissue_params: Tissue parameters [batch_size, 3]
        tissue_labels: Tissue name for each sample
    """
    if tissue_names is None:
        tissue_names = list(TISSUE_PARAMS.keys())

    n_acq = params.length
    signals = np.empty((batch_size, width, n_acq), dtype=np.complex128)
    tissue_params = np.empty((batch_size, 3))
    tissue_labels = []

    for i in range(batch_size):
        # Randomly select tissue type
        tissue_name = np.random.choice(tissue_names)
        tissue = TISSUE_PARAMS[tissue_name]

        # Get base T1/T2 with jitter
        T1 = tissue['t1'] * (1 + t1_t2_jitter * np.random.randn())
        T2 = tissue['t2'] * (1 + t1_t2_jitter * np.random.randn())
        T2 = min(T2, T1)  # Ensure T2 <= T1
        T2 = max(T2, 0.001)  # Ensure T2 > 0

        # Generate signal
        signal = generate_signal_simple(T1, T2, params, width, gradient, min_TR)

        # Add noise
        if noise_sigma > 0:
            noise = noise_sigma * (
                np.random.randn(*signal.shape) +
                1j * np.random.randn(*signal.shape)
            )
            signal = signal + noise

        signals[i] = signal
        tissue_params[i] = [T1, T2, T2 / T1]
        tissue_labels.append(tissue_name)

    return signals, tissue_params, tissue_labels


class ConditionalGASPDataset(Dataset):
    """
    PyTorch Dataset for Conditional GASP training.

    Generates data on-the-fly from simulation.
    """

    def __init__(
        self,
        params: SSFPParams,
        target_profile: npt.NDArray,
        n_samples: int = 10000,
        t1_range: tuple[float, float] = (0.1, 4.0),
        t2_t1_ratio_range: tuple[float, float] = (0.01, 0.5),
        width: int = 256,
        gradient: float = 2 * np.pi,
        min_TR: float = 5e-3,
        noise_sigma: float = 0.0,
        regenerate_each_epoch: bool = True,
    ):
        """
        Args:
            params: SSFPParams object with acquisition parameters
            target_profile: Target spectral profile [width]
            n_samples: Number of samples per epoch
            t1_range: (min, max) T1 values in seconds
            t2_t1_ratio_range: (min, max) T2/T1 ratio values
            width: Number of frequency points
            gradient: Maximum phase accumulation
            min_TR: Minimum TR for frequency calculation
            noise_sigma: Standard deviation of Gaussian noise
            regenerate_each_epoch: If True, regenerate data each epoch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ConditionalGASPDataset")

        self.params = params
        self.target_profile = target_profile
        self.n_samples = n_samples
        self.t1_range = t1_range
        self.t2_t1_ratio_range = t2_t1_ratio_range
        self.width = width
        self.gradient = gradient
        self.min_TR = min_TR
        self.noise_sigma = noise_sigma
        self.regenerate_each_epoch = regenerate_each_epoch

        # Generate initial data
        self._generate_data()

    def _generate_data(self):
        """Generate dataset."""
        self.signals, self.tissue_params = generate_training_batch(
            self.n_samples,
            self.params,
            self.t1_range,
            self.t2_t1_ratio_range,
            self.width,
            self.gradient,
            self.min_TR,
            noise_sigma=self.noise_sigma,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            signal: Complex signal [width, n_acquisitions]
            target: Target profile [width]
            tissue_params: [T1, T2, ratio]
        """
        signal = torch.from_numpy(self.signals[idx]).cfloat()
        target = torch.from_numpy(self.target_profile).float()
        tissue = torch.from_numpy(self.tissue_params[idx]).float()
        return signal, target, tissue

    def on_epoch_end(self):
        """Call at end of epoch to regenerate data if configured."""
        if self.regenerate_each_epoch:
            self._generate_data()


def create_dataloader(
    params: SSFPParams,
    target_profile: npt.NDArray,
    batch_size: int = 256,
    n_samples: int = 10000,
    num_workers: int = 0,
    **dataset_kwargs
) -> "DataLoader":
    """
    Create a DataLoader for Conditional GASP training.

    Args:
        params: SSFPParams object
        target_profile: Target spectral profile
        batch_size: Batch size
        n_samples: Samples per epoch
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for ConditionalGASPDataset

    Returns:
        DataLoader instance
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for create_dataloader")

    dataset = ConditionalGASPDataset(
        params, target_profile, n_samples, **dataset_kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
