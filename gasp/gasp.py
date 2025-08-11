"""GASP module"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from skimage.filters import threshold_li
from itertools import combinations

# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------

def _to_matrix(I: npt.NDArray) -> tuple[npt.NDArray, tuple[int, int]]:
    """Flatten [H, W, ...] → [H*W, features] and return original (H, W)."""
    if I.ndim < 3:
        raise ValueError("Expected I with shape [H, W, features...]")
    h, w = I.shape[:2]
    X = I.reshape(h * w, -1)
    return X, (h, w)

def _repeat_profile(D: npt.NDArray, n_samples: int) -> npt.NDArray:
    """Tile 1D desired profile D to length n_samples (exact multiple required)."""
    d = np.asarray(D).ravel()
    if d.size == n_samples:
        return d
    if n_samples % d.size != 0:
        raise ValueError(f"Cannot tile D of length {d.size} to {n_samples} samples.")
    return np.tile(d, n_samples // d.size)

def _design_matrix(X: npt.NDArray, method: str) -> npt.NDArray:
    """
    Build the design matrix Φ for the chosen method.
    Methods:
      - 'linear'      : Φ = [X]
      - 'affine'      : Φ = [1, X]
      - 'quad'        : Φ = [1, X, X^2]
      - 'quad-cross'  : Φ = [1, X, X^2, {X_i X_j}_{i<j}]
    """
    method = method.lower()
    n = X.shape[0]
    ones = np.ones((n, 1), dtype=X.dtype)

    if method == "linear":
        return X

    if method == "affine":
        return np.column_stack((ones, X))

    if method == "quad":
        return np.column_stack((ones, X, X**2))

    if method == "quad-cross":
        # Build crosses from the ORIGINAL predictors (not augmented or squared)
        p = X.shape[1]
        crosses = [(X[:, i] * X[:, j])[:, None] for i, j in combinations(range(p), 2)]
        cross_block = np.hstack(crosses) if crosses else np.empty((n, 0), dtype=X.dtype)
        return np.column_stack((ones, X, X**2, cross_block))

    raise ValueError(f"Unknown method '{method}'. Choose from "
                     f"{{'linear','affine','quad','quad-cross'}}.")

def l2_regularization(
    X: npt.NDArray,
    y: npt.NDArray,
    lam: float = 1e-2,
    *,
    penalise_bias: bool = False
) -> npt.NDArray:
    """
    Ridge regression: argmin_A ||X A - y||^2 + lam * ||A||^2.

    Handles real or complex X,y. If penalise_bias=False, the first column is not penalised.
    """
    n_terms = X.shape[1]
    L = np.eye(n_terms, dtype=X.dtype)
    if not penalise_bias:
        L[0, 0] = 0.0
    # Solve (X^H X + lam L) A = X^H y
    XtX = X.conj().T @ X
    Xty = X.conj().T @ y
    return np.linalg.solve(XtX + lam * L, Xty)

def _fit(
    Phi: npt.NDArray,
    D: npt.NDArray,
    useL2: bool = False,
    lam: float = 1e-2,
    *,
    penalise_bias: bool = False
) -> npt.NDArray:
    """Least-squares or ridge fit depending on flags."""
    if useL2:
        return l2_regularization(Phi, D, lam=lam, penalise_bias=penalise_bias)
    # lstsq handles real/complex and is numerically safer than solving normal eqs
    return np.linalg.lstsq(Phi, D, rcond=None)[0]

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def run_gasp(I: npt.NDArray, An: npt.NDArray, method: str = "affine") -> npt.NDArray:
    """
    Apply GASP model to data I using coefficients An.
    I: [H, W, PCs x TRs] (or more features collapsed)
    An: coefficient vector matching design matrix for 'method'
    Returns: image [H, W]
    """
    X, shape = _to_matrix(I)
    Phi = _design_matrix(X, method)
    out = (Phi @ An).reshape(shape)
    return out

def train_gasp(
    I: npt.NDArray,
    D: npt.NDArray,
    method: str = "affine",
    useL2: bool = False,
    lam: float = 1e-2,
    *,
    penalise_bias: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Train GASP coefficients from data I and desired profile D.

    I: [H, W, PCs x TRs]
    D: 1D vector sampled along the spectral dimension; will be tiled to match H*W
    method: 'linear' | 'affine' | 'quad' | 'quad-cross'
    useL2/lam: enable ridge (L2) regularization (useful especially for 'affine')
    penalise_bias: if True, the bias term (first column) is regularised too.

    Returns: (reconstruction [H, W], coefficients A)
    """
    X, shape = _to_matrix(I)
    Phi = _design_matrix(X, method)
    Dv = _repeat_profile(D, X.shape[0])
    A = _fit(Phi, Dv, useL2=useL2, lam=lam, penalise_bias=penalise_bias)
    out = (Phi @ A).reshape(shape)
    return out, np.asarray(A)

def train_gasp_with_coils(
    data: npt.NDArray,
    D: npt.NDArray,
    method: str = "affine",
    useL2: bool = False,
    lam: float = 1e-2,
    *,
    penalise_bias: bool = False
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Train per-coil GASP and combine with root-sum-of-squares (RSS).

    data: [H, W, coils, PCs] or [H, W, coils, PCs, TR]
    D: desired 1D profile
    Returns: (RSS image [H, W], per-coil coefficients [coils, n_terms])
    """
    if data.ndim == 4:
        h, w, ncoils, npcs = data.shape
        Xc = data
    elif data.ndim == 5:
        h, w, ncoils, pcs, TRs = data.shape
        Xc = data.reshape(h, w, ncoils, pcs * TRs)
    else:
        raise ValueError("Expected data with shape [H, W, coils, PCs] or [H, W, coils, PCs, TR]")

    outs = np.zeros((ncoils, h, w), dtype=complex)
    A_list = []
    for c in range(ncoils):
        out_c, A_c = train_gasp(
            Xc[:, :, c, :],
            D,
            method=method,
            useL2=useL2,
            lam=lam,
            penalise_bias=penalise_bias,
        )
        outs[c] = out_c
        A_list.append(np.asarray(A_c))
    An = np.vstack(A_list)
    rss = np.sqrt(np.sum(np.abs(outs) ** 2, axis=0))
    return rss, An

# ---------------------------------------------------------------------
# Pre/post-processing helpers
# ---------------------------------------------------------------------

def create_data_mask(M: npt.NDArray) -> npt.NDArray:
    """
    Create a foreground mask using Li threshold on a magnitude summary.
    Works for M with shape [H, W, ...]. Reduces all trailing dims by mean(|·|).
    """
    mag = np.abs(M)
    while mag.ndim > 2:
        mag = mag.mean(axis=-1)
    mag = np.asarray(mag, dtype=float)
    thresh = threshold_li(mag)
    return mag > thresh

def apply_mask_to_data(M: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    """
    Broadcast-mask M with mask [H, W]. Returns masked array with same shape as M.
    """
    if mask.shape != M.shape[:2]:
        raise ValueError("Mask must match the first two dims of M (H, W).")
    m = mask.astype(bool)
    # Broadcast across trailing dims
    while m.ndim < M.ndim:
        m = m[..., None]
    return M * m

def extract_centered_subset(data: npt.NDArray, n_lines: int) -> npt.NDArray:
    """
    Extract n_lines around the vertical (height) center: data[start:end, ...]
    """
    if n_lines > data.shape[0]:
        raise ValueError("n_lines cannot exceed data height.")
    center = data.shape[0] // 2
    start = center - n_lines // 2
    end = start + n_lines
    return data[start:end, ...]

def process_data_for_gasp(
    M: npt.NDArray,
    useMask: bool = False,
    useCalibration: bool = False,
    n_lines: int = 2
) -> npt.NDArray:
    """
    Optional pre-processing: foreground mask and/or a central calibration strip.
    """
    data = M
    if useMask:
        mask = create_data_mask(M)
        data = apply_mask_to_data(data, mask)
    if useCalibration:
        data = extract_centered_subset(data, n_lines)
    return data
