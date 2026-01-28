"""Amplitude compensation for universal GASP coefficients."""

from .amplitude_compensation import (
    estimate_t2_from_tr_decay,
    estimate_t1_from_profile,
    estimate_t1_t2_planet,
    compute_ssfp_amplitude,
    normalize_signals,
    train_compensated_gasp,
    run_compensated_gasp,
)

__all__ = [
    "estimate_t2_from_tr_decay",
    "estimate_t1_from_profile",
    "estimate_t1_t2_planet",
    "compute_ssfp_amplitude",
    "normalize_signals",
    "train_compensated_gasp",
    "run_compensated_gasp",
]
