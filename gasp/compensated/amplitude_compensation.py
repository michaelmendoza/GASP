"""T2/T1 estimation and amplitude compensation for universal GASP."""

import numpy as np
import numpy.typing as npt


# =============================================================================
# PLANET-based T1/T2 Estimation
# =============================================================================

def fit_ellipse(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit an ellipse to 2D points using direct least squares.

    Uses the algebraic ellipse equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    with constraint B² - 4AC < 0 (ensures ellipse, not hyperbola).

    Returns (a, b, cx, cy, angle) - semi-axes, center, rotation angle.
    """
    # Build design matrix for general conic: [x², xy, y², x, y, 1]
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])

    # Constraint matrix for ellipse (B² - 4AC < 0)
    # Using Fitzgibbon's method with constraint C1 = [0 0 2; 0 -1 0; 2 0 0]
    S = D.T @ D

    # Partition S into blocks
    S1 = S[:3, :3]
    S2 = S[:3, 3:]
    S3 = S[3:, 3:]

    # Constraint matrix
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    # Solve generalized eigenvalue problem
    try:
        S3_inv = np.linalg.inv(S3)
        M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)
        eigenvalues, eigenvectors = np.linalg.eig(M)

        # Find the positive eigenvalue (ellipse condition)
        cond = 4 * eigenvectors[0] * eigenvectors[2] - eigenvectors[1]**2
        idx = np.where(cond > 0)[0]
        if len(idx) == 0:
            return None

        # Choose eigenvector with smallest positive eigenvalue
        a1 = eigenvectors[:, idx[np.argmin(np.abs(eigenvalues[idx]))]]
        a2 = -S3_inv @ S2.T @ a1

        coeffs = np.concatenate([a1, a2])
        A, B, C, D, E, F = coeffs

        # Convert to geometric parameters
        # Center
        denom = B**2 - 4*A*C
        if abs(denom) < 1e-10:
            return None
        cx = (2*C*D - B*E) / denom
        cy = (2*A*E - B*D) / denom

        # Rotation angle
        angle = 0.5 * np.arctan2(B, A - C)

        # Semi-axes
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        A_rot = A*cos_a**2 + B*cos_a*sin_a + C*sin_a**2
        C_rot = A*sin_a**2 - B*cos_a*sin_a + C*cos_a**2

        F_centered = F + A*cx**2 + B*cx*cy + C*cy**2 + D*cx + E*cy

        if abs(A_rot) < 1e-10 or abs(C_rot) < 1e-10 or F_centered >= 0:
            return None

        a = np.sqrt(-F_centered / A_rot)
        b = np.sqrt(-F_centered / C_rot)

        # Ensure a >= b (a is semi-major axis)
        if a < b:
            a, b = b, a
            angle = angle + np.pi/2

        return (a, b, cx, cy, angle)
    except (np.linalg.LinAlgError, ValueError):
        return None


def fit_ellipse_algebraic(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """
    Fit ellipse and return algebraic coefficients (A, B, C, D, E, F).

    Ellipse equation: Ax² + Bxy + Cy² + Dx + Ey + F = 0
    """
    # Build design matrix
    D = np.column_stack([x**2, x*y, y**2, x, y, np.ones_like(x)])
    S = D.T @ D

    # Partition and solve
    S1, S2 = S[:3, :3], S[:3, 3:]
    S3 = S[3:, 3:]
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)

    try:
        S3_inv = np.linalg.inv(S3)
        M = np.linalg.inv(C1) @ (S1 - S2 @ S3_inv @ S2.T)
        eigenvalues, eigenvectors = np.linalg.eig(M)

        cond = 4 * eigenvectors[0] * eigenvectors[2] - eigenvectors[1]**2
        idx = np.where(cond > 0)[0]
        if len(idx) == 0:
            return None

        a1 = eigenvectors[:, idx[np.argmin(np.abs(eigenvalues[idx]))]]
        a2 = -S3_inv @ S2.T @ a1

        return np.real(np.concatenate([a1, a2]))
    except:
        return None


def estimate_t1_t2_planet(
    signals: npt.NDArray,
    TR: float,
    alpha: float,
    npcs: int = 8,
    T1_guess: float = 1.0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Estimate T1, T2, and off-resonance using PLANET algorithm.

    Based on mckib2/ssfp implementation of Shcherbakova et al., MRM 2018.

    Parameters
    ----------
    signals : ndarray [..., npcs]
        Phase-cycled bSSFP signals for a single TR
    TR : float
        Repetition time in seconds
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles (minimum 6 recommended)
    T1_guess : float
        Initial T1 estimate for sign determination

    Returns
    -------
    t1_map, t2_map, f0_map : ndarrays
    """
    shape = signals.shape[:-1]
    flat_shape = int(np.prod(shape)) if shape else 1
    signals_flat = signals.reshape(flat_shape, npcs)

    t1_flat = np.ones(flat_shape) * T1_guess
    t2_flat = np.ones(flat_shape) * 0.1
    f0_flat = np.zeros(flat_shape)

    cos_alpha = np.cos(alpha)

    for i in range(flat_shape):
        sig = signals_flat[i]
        x_data = np.real(sig)
        y_data = np.imag(sig)

        # Fit ellipse - get algebraic coefficients
        coeffs = fit_ellipse_algebraic(x_data, y_data)
        if coeffs is None:
            continue

        A, B, C, D, E, F = coeffs

        # Find ellipse rotation angle phi
        phi = 0.5 * np.arctan2(B, A - C)

        c, s = np.cos(phi), np.sin(phi)
        c2, s2 = c**2, s**2

        # Rotate ellipse to align with axes
        A1 = A*c2 + B*c*s + C*s2
        D1 = D*c + E*s
        E1 = E*c - D*s
        C1 = A*s2 - B*c*s + C*c2

        if abs(A1) < 1e-10 or abs(C1) < 1e-10:
            continue

        F11 = F - ((D1**2)/(4*A1) + (E1**2)/(4*C1))

        # Ellipse center and semi-axes
        Xc = -D1/(2*A1)
        Yc = -E1/(2*C1)

        if F11/A1 > 0 or F11/C1 > 0:
            continue

        aa = np.sqrt(-F11/A1)  # semi-axis along rotated x
        bb = np.sqrt(-F11/C1)  # semi-axis along rotated y

        # Adjust phi based on ellipse orientation (from reference implementation)
        aa_le_bb = aa <= bb
        if aa_le_bb and Xc < 0:
            phi -= np.pi * np.sign(phi) if phi != 0 else np.pi
        elif not aa_le_bb and Yc >= 0:
            phi += np.pi/2
        elif not aa_le_bb and Yc < 0:
            phi -= np.pi/2

        # Recompute after phi adjustment
        if aa_le_bb and Xc < 0 or not aa_le_bb:
            c, s = np.cos(phi), np.sin(phi)
            c2, s2 = c**2, s**2
            A1 = A*c2 + B*c*s + C*s2
            D1 = D*c + E*s
            E1 = E*c - D*s
            C1 = A*s2 - B*c*s + C*c2

            if abs(A1) < 1e-10 or abs(C1) < 1e-10:
                continue

            F11 = F - ((D1**2)/(4*A1) + (E1**2)/(4*C1))
            Xc = -D1/(2*A1)
            Yc = -E1/(2*C1)

            if F11/A1 > 0 or F11/C1 > 0:
                continue

            aa = np.sqrt(-F11/A1)
            bb = np.sqrt(-F11/C1)

        # Decide sign of b based on flip angle
        if alpha > np.arccos(np.exp(-TR/T1_guess)):
            bsign = -1
        else:
            bsign = 1

        # Compute PLANET parameters a and b (key equations from reference)
        Xc2 = Xc * Xc
        bb2 = bb * bb

        discriminant = Xc2 - aa*aa + bb2
        if discriminant < 0:
            continue

        b = (bsign * Xc * aa + bb * np.sqrt(discriminant)) / (Xc2 + bb2)
        b2 = b * b

        if b2 >= 1:
            continue

        denom = b * bb + Xc * np.sqrt(1 - b2)
        if abs(denom) < 1e-10:
            continue

        a = bb / denom

        # Validate a is in valid range for log
        if a <= 0 or a >= 1:
            continue

        # T2 = -TR / ln(a)
        t2_flat[i] = -TR / np.log(a)

        # T1 from PLANET equation
        ab = a * b
        numer = a * (1 + cos_alpha - ab * cos_alpha) - b
        denom_t1 = a * (1 + cos_alpha - ab) - b * cos_alpha

        if abs(denom_t1) < 1e-10 or numer/denom_t1 <= 0 or numer/denom_t1 >= 1:
            t1_flat[i] = T1_guess
        else:
            t1_flat[i] = -TR / np.log(numer / denom_t1)

        # Off-resonance from rotation angle
        f0_flat[i] = phi / (2 * np.pi * TR)

    # Clip to reasonable ranges
    t1_flat = np.clip(t1_flat, 0.05, 10.0)
    t2_flat = np.clip(t2_flat, 0.001, 5.0)

    return (
        t1_flat.reshape(shape) if shape else t1_flat[0],
        t2_flat.reshape(shape) if shape else t2_flat[0],
        f0_flat.reshape(shape) if shape else f0_flat[0],
    )


def estimate_t2_t1_ratio(
    signals: npt.NDArray,
    TR: float,
    alpha: float,
    npcs: int = 8,
) -> npt.NDArray:
    """
    Estimate T2/T1 ratio directly from bSSFP signal characteristics.

    This is more robust than estimating T1 and T2 separately for
    amplitude compensation purposes.

    Uses the on-resonance to off-resonance signal ratio which encodes T2/T1.

    Parameters
    ----------
    signals : ndarray [..., npcs]
        Phase-cycled bSSFP signals
    TR : float
        Repetition time in seconds
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles

    Returns
    -------
    ratio : ndarray [...]
        Estimated T2/T1 ratio
    """
    shape = signals.shape[:-1]

    # Get magnitude of signals at different phase cycles
    mag = np.abs(signals)

    # On-resonance (phase cycle 0) vs off-resonance (phase cycle at π)
    # The ratio of these encodes information about E1*E2
    S_on = mag[..., 0]
    S_off = mag[..., npcs // 2]

    eps = 1e-10

    # Signal ratio: r = S_on / S_off
    # For bSSFP: r ≈ (1 + E1*E2) / (1 - E1*E2)
    # Solving: E1*E2 = (r - 1) / (r + 1)
    r = S_on / (S_off + eps)
    E1E2 = np.clip((r - 1) / (r + 1 + eps), 0.01, 0.99)

    # E1*E2 = exp(-TR/T1) * exp(-TR/T2) = exp(-TR * (1/T1 + 1/T2))
    # For T2/T1 = k, we have T2 = k*T1, so:
    # E1*E2 = exp(-TR/T1 * (1 + 1/k))
    #
    # Also use signal amplitude to constrain:
    # Mean signal amplitude relates to T2/T1 through bSSFP equation

    # Use ellipse eccentricity as additional constraint
    x = np.real(signals)
    y = np.imag(signals)

    # Compute signal spread (proxy for ellipse shape)
    x_range = np.max(x, axis=-1) - np.min(x, axis=-1)
    y_range = np.max(y, axis=-1) - np.min(y, axis=-1)

    # Aspect ratio of signal distribution
    aspect = np.minimum(x_range, y_range) / (np.maximum(x_range, y_range) + eps)

    # Map E1*E2 and aspect to T2/T1 ratio
    # Higher E1*E2 (closer to 1) -> longer relaxation times
    # Higher aspect (rounder) -> larger T2/T1 ratio

    # Empirical mapping based on bSSFP behavior:
    # E1*E2 ≈ exp(-2*TR/T2) when T2 << T1 (typical tissue)
    # So T2 ≈ -2*TR / ln(E1*E2) as rough estimate

    # T2/T1 ratio typically ranges from 0.01 (short T2 tissues) to 0.5 (fluids)
    # Use combination of E1E2 and aspect ratio
    ratio = 0.05 + 0.45 * aspect * np.sqrt(E1E2)

    return np.clip(ratio, 0.01, 1.0)


def _compute_ssfp_amplitude_internal(T1, T2, TR, alpha):
    """Internal helper to compute bSSFP amplitude (avoids circular import)."""
    E1 = np.exp(-TR / np.maximum(T1, 1e-10))
    E2 = np.exp(-TR / np.maximum(T2, 1e-10))
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    numerator = sin_alpha * (1 - E1)
    denominator = 1 - E1 * E2 - (E1 - E2) * cos_alpha
    return numerator / np.maximum(np.abs(denominator), 1e-10)


def estimate_t2_from_tr_decay(
    signals: npt.NDArray,
    TRs: list[float],
    npcs: int = 16,
    alpha: float = np.deg2rad(60),
    t1_assumed: float = 1.0,
) -> npt.NDArray:
    """
    Estimate T2 using dictionary matching against bSSFP signal model.

    The bSSFP steady-state signal does NOT follow simple exp(-TR/T2) decay.
    Instead, we use the full bSSFP equation and find the T2 that best
    matches the observed signal ratios across TRs.

    Parameters
    ----------
    signals : ndarray [..., npcs * nTRs]
        Multi-TR phase-cycled signals (last dim is flattened PCs×TRs)
    TRs : list[float]
        Repetition times in seconds (e.g., [5e-3, 10e-3, 15e-3])
    npcs : int
        Number of phase cycles per TR
    alpha : float
        Flip angle in radians (needed for bSSFP model)
    t1_assumed : float
        Assumed T1 for estimation (default 1.0s)

    Returns
    -------
    t2_estimate : ndarray [...]
        Estimated T2 in seconds
    """
    # Reshape to separate TRs
    shape = signals.shape[:-1]
    nTRs = len(TRs)
    signals_reshaped = signals.reshape(*shape, npcs, nTRs)

    # Mean magnitude per TR (average over phase cycles)
    mag_per_tr = np.abs(signals_reshaped).mean(axis=-2)  # [..., nTRs]

    # Normalize to first TR to get relative signal ratios
    mag_ratio = mag_per_tr / (mag_per_tr[..., 0:1] + 1e-10)  # [..., nTRs]

    # Create dictionary of T2 candidates (log-spaced for better coverage)
    t2_candidates = np.logspace(-3, 1, 200)  # 1ms to 10s

    # Compute theoretical signal ratios for each T2 candidate
    # Shape: [n_candidates, nTRs]
    theoretical_ratios = np.zeros((len(t2_candidates), nTRs))
    for i, t2 in enumerate(t2_candidates):
        amps = np.array([_compute_ssfp_amplitude_internal(t1_assumed, t2, tr, alpha) for tr in TRs])
        theoretical_ratios[i] = amps / (amps[0] + 1e-10)

    # Find best matching T2 for each voxel using least squares
    # Flatten spatial dimensions for vectorized computation
    flat_shape = np.prod(shape) if shape else 1
    mag_ratio_flat = mag_ratio.reshape(flat_shape, nTRs)

    # Compute squared error for all voxels vs all candidates
    # Shape: [n_voxels, n_candidates]
    errors = np.sum((mag_ratio_flat[:, np.newaxis, :] - theoretical_ratios[np.newaxis, :, :]) ** 2, axis=-1)

    # Find best T2 for each voxel
    best_idx = np.argmin(errors, axis=-1)
    t2_estimate = t2_candidates[best_idx]

    return t2_estimate.reshape(shape) if shape else t2_estimate


def estimate_t1_from_profile(
    signals: npt.NDArray,
    t2_estimate: npt.NDArray,
    TRs: list[float],
    alpha: float,
    npcs: int = 16,
) -> npt.NDArray:
    """
    Estimate T1 from bSSFP profile shape, given T2 estimate.

    The ratio of on-resonance (θ≈0) to off-resonance (θ≈π) signal
    depends on T2/T1. Given T2, we can solve for T1.

    For bSSFP at on-resonance:  S_on  ∝ (1-E1)/(1-E1*E2)
    For bSSFP at off-resonance: S_off ∝ (1-E1)/(1+E1*E2)

    Ratio: S_on/S_off ≈ (1+E1*E2)/(1-E1*E2)

    Parameters
    ----------
    signals : ndarray [..., npcs * nTRs]
        Multi-TR phase-cycled signals
    t2_estimate : ndarray [...]
        Previously estimated T2 values
    TRs : list[float]
        Repetition times in seconds
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles per TR

    Returns
    -------
    t1_estimate : ndarray [...]
        Estimated T1 in seconds
    """
    shape = signals.shape[:-1]
    nTRs = len(TRs)
    signals_reshaped = signals.reshape(*shape, npcs, nTRs)

    # Use first TR for profile analysis (highest SNR typically)
    profile = np.abs(signals_reshaped[..., 0])  # [..., npcs]

    # On-resonance is at phase cycle 0, off-resonance at npcs//2 (θ=π)
    S_on = profile[..., 0]
    S_off = profile[..., npcs // 2]

    eps = 1e-10
    ratio = S_on / (S_off + eps)

    # For bSSFP: ratio ≈ (1 + E1*E2) / (1 - E1*E2)
    # Solve for E1*E2: let r = ratio
    # r*(1 - E1*E2) = 1 + E1*E2
    # r - r*E1*E2 = 1 + E1*E2
    # r - 1 = E1*E2*(r + 1)
    # E1*E2 = (r - 1) / (r + 1)

    TR = TRs[0]
    E2 = np.exp(-TR / t2_estimate)
    E1_E2_product = (ratio - 1) / (ratio + 1 + eps)
    E1_E2_product = np.clip(E1_E2_product, eps, 1.0 - eps)

    E1 = E1_E2_product / (E2 + eps)
    E1 = np.clip(E1, eps, 1.0 - eps)

    # T1 = -TR / ln(E1)
    t1_estimate = -TR / np.log(E1)

    return np.clip(t1_estimate, 0.1, 5.0)  # Clamp to reasonable range


def compute_ssfp_amplitude(
    T1: npt.NDArray | float,
    T2: npt.NDArray | float,
    TR: float,
    alpha: float,
) -> npt.NDArray:
    """
    Compute theoretical on-resonance bSSFP amplitude.

    A = sin(α)·(1-E1) / (1 - E1·E2 - (E1-E2)·cos(α))
    """
    E1 = np.exp(-TR / np.maximum(T1, 1e-10))
    E2 = np.exp(-TR / np.maximum(T2, 1e-10))

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    numerator = sin_alpha * (1 - E1)
    denominator = 1 - E1 * E2 - (E1 - E2) * cos_alpha

    return numerator / np.maximum(np.abs(denominator), 1e-10)


def normalize_signals(
    signals: npt.NDArray,
    TRs: list[float],
    alpha: float,
    npcs: int = 16,
    reference_t2_t1: float = 0.1,
    t1_reference: float = 1.0,
    estimate_t1: bool = False,
    known_t1: float | None = None,
    known_t2: float | None = None,
) -> tuple[npt.NDArray, dict]:
    """
    Normalize signals to compensate for T2/T1-dependent amplitude.

    Parameters
    ----------
    signals : ndarray [..., npcs * nTRs]
        Multi-TR phase-cycled signals
    TRs : list[float]
        Repetition times in seconds
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles per TR
    reference_t2_t1 : float
        Target T2/T1 ratio for normalization (default: 0.1)
    t1_reference : float
        Reference T1 value when estimate_t1=False (default: 1.0s)
    estimate_t1 : bool
        If True, estimate T1 from profile shape after T2 estimation.
        If False, use fixed t1_reference value.
    known_t1 : float, optional
        If provided, use this T1 value instead of estimating (for simulations).
    known_t2 : float, optional
        If provided, use this T2 value instead of estimating (for simulations).

    Returns
    -------
    normalized : ndarray
        Amplitude-compensated signals
    estimates : dict
        Contains 't2': T2 map, 't1': T1 map (or reference value)
    """
    # Use known values if provided, otherwise estimate
    if known_t2 is not None:
        t2_estimate = known_t2
    else:
        t2_estimate = estimate_t2_from_tr_decay(signals, TRs, npcs, alpha, t1_reference)

    if known_t1 is not None:
        t1_estimate = known_t1
    elif estimate_t1:
        t1_estimate = estimate_t1_from_profile(signals, t2_estimate, TRs, alpha, npcs)
    else:
        t1_estimate = t1_reference  # Use fixed value

    # Reference values for normalization target
    t2_ref = reference_t2_t1 * t1_reference

    # Compute per-TR normalization factors
    shape = signals.shape[:-1]
    nTRs = len(TRs)
    signals_reshaped = signals.reshape(*shape, npcs, nTRs)
    normalized_blocks = []

    for i, TR in enumerate(TRs):
        # Amplitude at estimated T1, T2
        amp_actual = compute_ssfp_amplitude(t1_estimate, t2_estimate, TR, alpha)
        # Amplitude at reference T2/T1
        amp_ref = compute_ssfp_amplitude(t1_reference, t2_ref, TR, alpha)

        # Scale factor: multiply by (ref / actual)
        scale = amp_ref / (amp_actual + 1e-10)
        if isinstance(scale, np.ndarray):
            scale = scale[..., np.newaxis]  # Add PC dimension

        normalized_blocks.append(signals_reshaped[..., i] * scale)

    normalized = np.stack(normalized_blocks, axis=-1)
    estimates = {'t2': t2_estimate, 't1': t1_estimate}
    return normalized.reshape(signals.shape), estimates


# High-level API
def train_compensated_gasp(
    I: npt.NDArray,
    D: npt.NDArray,
    TRs: list[float],
    alpha: float,
    npcs: int = 16,
    method: str = "affine",
    reference_t2_t1: float = 0.1,
    estimate_t1: bool = False,
    known_t1: float | None = None,
    known_t2: float | None = None,
    **gasp_kwargs
):
    """
    Train GASP with automatic amplitude compensation.

    Parameters
    ----------
    I : ndarray [..., npcs * nTRs]
        Multi-TR phase-cycled signals
    D : ndarray
        Desired spectral profile
    TRs : list[float]
        Repetition times in seconds
    alpha : float
        Flip angle in radians
    npcs : int
        Number of phase cycles per TR
    method : str
        GASP method ('linear', 'affine', 'quad', 'quad-cross')
    reference_t2_t1 : float
        Target T2/T1 for normalization
    estimate_t1 : bool
        If True, estimate T1 from profile shape.
        If False, assume T1=1.0s (faster, often sufficient).
    known_t1 : float, optional
        If provided, use this T1 value instead of estimating.
    known_t2 : float, optional
        If provided, use this T2 value instead of estimating.
    **gasp_kwargs
        Additional arguments passed to train_gasp

    Returns
    -------
    reconstruction, coefficients, metadata
    """
    from ..gasp import train_gasp

    I_norm, estimates = normalize_signals(
        I, TRs, alpha, npcs, reference_t2_t1,
        estimate_t1=estimate_t1,
        known_t1=known_t1,
        known_t2=known_t2,
    )
    reconstruction, coefficients = train_gasp(I_norm, D, method=method, **gasp_kwargs)

    return reconstruction, coefficients, {
        't2_estimate': estimates['t2'],
        't1_estimate': estimates['t1'],
        'reference_t2_t1': reference_t2_t1,
        'estimate_t1': estimate_t1,
        'TRs': TRs,
        'alpha': alpha,
        'npcs': npcs,
    }


def run_compensated_gasp(
    I: npt.NDArray,
    coefficients: npt.NDArray,
    metadata: dict,
    method: str = "affine",
    known_t1: float | None = None,
    known_t2: float | None = None,
    denormalize: bool = False,
):
    """
    Apply GASP with automatic amplitude compensation.

    Parameters
    ----------
    I : ndarray
        Input signals
    coefficients : ndarray
        GASP coefficients from training
    metadata : dict
        Metadata from train_compensated_gasp
    method : str
        GASP method
    known_t1 : float, optional
        Known T1 value for the test tissue (for simulations)
    known_t2 : float, optional
        Known T2 value for the test tissue (for simulations)
    denormalize : bool
        If True, scale output to restore T2/T1-based amplitude contrast.
        This gives accurate spectral shaping while preserving tissue contrast.
        Default: False (output matches normalized reference amplitude)
    """
    from ..gasp import run_gasp

    # Get T1/T2 for this tissue (needed for denormalization)
    t1 = known_t1 if known_t1 is not None else metadata.get('t1_estimate', 1.0)
    t2 = known_t2 if known_t2 is not None else metadata.get('t2_estimate', 0.1)

    I_norm, _ = normalize_signals(
        I,
        metadata['TRs'],
        metadata['alpha'],
        metadata['npcs'],
        metadata['reference_t2_t1'],
        estimate_t1=metadata.get('estimate_t1', False),
        known_t1=known_t1,
        known_t2=known_t2,
    )

    output = run_gasp(I_norm, coefficients, method=method)

    # Optionally denormalize to restore T2/T1-based contrast
    if denormalize:
        TR = metadata['TRs'][0]  # Use first TR for amplitude calculation
        alpha = metadata['alpha']
        t1_ref = 1.0  # Reference T1
        t2_ref = metadata['reference_t2_t1'] * t1_ref

        # Compute amplitude ratio: actual / reference
        amp_actual = compute_ssfp_amplitude(t1, t2, TR, alpha)
        amp_ref = compute_ssfp_amplitude(t1_ref, t2_ref, TR, alpha)

        # Scale output to restore original amplitude
        denorm_scale = amp_actual / (amp_ref + 1e-10)
        output = output * denorm_scale

    return output
