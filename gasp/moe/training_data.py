"""Training data generation for MoE-GASP.

This module provides utilities for generating diverse synthetic SSFP training
data with known T2/T1 ratios, which is essential for training the Mixture of
Experts GASP model.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from gasp.ssfp import ssfp, add_noise


def sample_tissue_parameters(
    n_samples: int,
    t2_t1_range: tuple[float, float] = (0.001, 0.5),
    t1_range: tuple[float, float] = (0.2, 2.0),
    f0_range: tuple[float, float] = (-200, 200),
    seed: int | None = None
) -> dict[str, npt.NDArray]:
    """
    Sample tissue parameters covering the T2/T1 space.

    Uses log-uniform sampling for T2/T1 ratios to ensure better coverage
    of short T2 tissues which are often underrepresented.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    t2_t1_range : tuple[float, float]
        Range of T2/T1 ratios to sample (log-uniform).
    t1_range : tuple[float, float]
        Range of T1 values in seconds (uniform).
    f0_range : tuple[float, float]
        Range of off-resonance frequencies in Hz (uniform).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, ndarray]
        Dictionary with keys: 't1', 't2', 't2_t1_ratio', 'f0'.
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
    t1_range: tuple[float, float] = (0.2, 2.0),
    f0_range: tuple[float, float] = (-200, 200),
    add_noise_flag: bool = True,
    noise_sigma: float = 0.01,
    seed: int | None = None
) -> tuple[npt.NDArray, npt.NDArray, dict]:
    """
    Generate SSFP signals for training MoE-GASP.

    Simulates SSFP signals across a diverse range of tissue parameters,
    providing training data that covers different T2/T1 regimes.

    Parameters
    ----------
    n_samples : int
        Number of training samples to generate.
    TR : float or list[float]
        Repetition time(s) in seconds. Can be single value or list for
        multi-TR acquisition.
    TE : float, optional
        Echo time in seconds. Defaults to TR/2.
    alpha : float
        Flip angle in radians.
    npcs : int
        Number of phase cycles.
    t2_t1_range : tuple[float, float]
        Range of T2/T1 ratios to sample.
    t1_range : tuple[float, float]
        Range of T1 values in seconds.
    f0_range : tuple[float, float]
        Range of off-resonance frequencies in Hz.
    add_noise_flag : bool
        Whether to add Gaussian noise to signals.
    noise_sigma : float
        Standard deviation of noise to add.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    signals : ndarray [n_samples, n_features]
        SSFP signals (flattened across phase cycles and TRs).
    t2_t1_ratios : ndarray [n_samples]
        True T2/T1 ratio for each sample.
    params : dict
        Full parameter dictionary including 't1', 't2', 't2_t1_ratio', 'f0'.
    """
    TRs = np.atleast_1d(TR)

    # Sample tissue parameters
    params = sample_tissue_parameters(
        n_samples,
        t2_t1_range=t2_t1_range,
        t1_range=t1_range,
        f0_range=f0_range,
        seed=seed
    )

    # Phase cycle increments
    dphi = np.linspace(0, 2 * np.pi, npcs, endpoint=False)

    # Generate signals
    signals = []
    for i in range(n_samples):
        sample_signals = []
        for tr in TRs:
            te = tr / 2 if TE is None else TE
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
    t2_t1_ratios: npt.NDArray,
    n_regimes: int = 5,
    method: str = 'quantile'
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Partition samples into T2/T1 regimes for expert assignment.

    Parameters
    ----------
    t2_t1_ratios : ndarray
        T2/T1 ratios for each sample.
    n_regimes : int
        Number of regimes/experts to create.
    method : str
        Partitioning method:
        - 'quantile': Equal number of samples per regime.
        - 'uniform': Equal-width bins in log space.

    Returns
    -------
    regime_labels : ndarray [n_samples]
        Regime index (0 to n_regimes-1) for each sample.
    regime_edges : ndarray [n_regimes + 1]
        Bin edges defining regime boundaries.
    """
    if method == 'quantile':
        # Equal number of samples per regime
        percentiles = np.linspace(0, 100, n_regimes + 1)
        regime_edges = np.percentile(t2_t1_ratios, percentiles)
    elif method == 'uniform':
        # Equal-width bins in log space
        log_ratios = np.log(t2_t1_ratios)
        regime_edges = np.exp(
            np.linspace(log_ratios.min(), log_ratios.max(), n_regimes + 1)
        )
    else:
        raise ValueError(f"Unknown partitioning method: {method}")

    # Assign labels using digitize
    # digitize returns 0 for values below first edge, so we use edges[1:-1]
    regime_labels = np.digitize(t2_t1_ratios, regime_edges[1:-1])

    return regime_labels, regime_edges


def generate_stratified_training_data(
    n_samples_per_regime: int,
    n_regimes: int = 5,
    TR: float | list[float] = 0.005,
    alpha: float = np.deg2rad(30),
    npcs: int = 16,
    t2_t1_range: tuple[float, float] = (0.001, 0.5),
    add_noise_flag: bool = True,
    noise_sigma: float = 0.01,
    seed: int | None = None
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, dict]:
    """
    Generate stratified training data with equal samples per T2/T1 regime.

    This ensures balanced representation across all T2/T1 regimes, which is
    important for training effective expert models.

    Parameters
    ----------
    n_samples_per_regime : int
        Number of samples to generate per regime.
    n_regimes : int
        Number of T2/T1 regimes.
    TR : float or list[float]
        Repetition time(s) in seconds.
    alpha : float
        Flip angle in radians.
    npcs : int
        Number of phase cycles.
    t2_t1_range : tuple[float, float]
        Overall range of T2/T1 ratios.
    add_noise_flag : bool
        Whether to add noise.
    noise_sigma : float
        Noise standard deviation.
    seed : int, optional
        Random seed.

    Returns
    -------
    signals : ndarray [n_samples, n_features]
        SSFP signals.
    t2_t1_ratios : ndarray [n_samples]
        T2/T1 ratios.
    regime_labels : ndarray [n_samples]
        Pre-assigned regime labels.
    params : dict
        Full parameter dictionary.
    """
    rng = np.random.default_rng(seed)

    # Define regime edges in log space
    log_edges = np.linspace(
        np.log(t2_t1_range[0]),
        np.log(t2_t1_range[1]),
        n_regimes + 1
    )
    regime_edges = np.exp(log_edges)

    all_signals = []
    all_t2_t1 = []
    all_labels = []
    all_params = {'t1': [], 't2': [], 't2_t1_ratio': [], 'f0': []}

    TRs = np.atleast_1d(TR)
    dphi = np.linspace(0, 2 * np.pi, npcs, endpoint=False)

    for regime_idx in range(n_regimes):
        # Sample T2/T1 within this regime's range
        regime_t2_t1_range = (regime_edges[regime_idx], regime_edges[regime_idx + 1])

        params = sample_tissue_parameters(
            n_samples_per_regime,
            t2_t1_range=regime_t2_t1_range,
            seed=rng.integers(0, 2**31) if seed is not None else None
        )

        # Generate signals for this regime
        for i in range(n_samples_per_regime):
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

            all_signals.append(np.concatenate(sample_signals))
            all_t2_t1.append(params['t2_t1_ratio'][i])
            all_labels.append(regime_idx)

            for key in all_params:
                all_params[key].append(params[key][i])

    # Convert to arrays
    signals = np.array(all_signals)
    t2_t1_ratios = np.array(all_t2_t1)
    regime_labels = np.array(all_labels)
    for key in all_params:
        all_params[key] = np.array(all_params[key])

    # Shuffle data
    shuffle_idx = rng.permutation(len(signals))
    signals = signals[shuffle_idx]
    t2_t1_ratios = t2_t1_ratios[shuffle_idx]
    regime_labels = regime_labels[shuffle_idx]
    for key in all_params:
        all_params[key] = all_params[key][shuffle_idx]

    return signals, t2_t1_ratios, regime_labels, all_params
