"""Dictionary Learning / Sparse Coding for GASP."""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit


def _to_matrix(I: npt.NDArray) -> tuple[npt.NDArray, tuple[int, int]]:
    """Flatten [H, W, ...] -> [H*W, features] and return original (H, W)."""
    if I.ndim < 3:
        raise ValueError("Expected I with shape [H, W, features...]")
    h, w = I.shape[:2]
    X = I.reshape(h * w, -1)
    return X, (h, w)


def _repeat_profile(D: npt.NDArray, n_samples: int) -> npt.NDArray:
    """Tile 1D desired profile D to length n_samples."""
    d = np.asarray(D).ravel()
    if d.size == n_samples:
        return d
    if n_samples % d.size != 0:
        raise ValueError(f"Cannot tile D of length {d.size} to {n_samples} samples.")
    return np.tile(d, n_samples // d.size)


def learn_dictionary(
    X: npt.NDArray,
    n_atoms: int = 16,
    alpha: float = 1.0,
    max_iter: int = 1000,
    batch_size: int = 256,
    random_state: int | None = None,
) -> npt.NDArray:
    """
    Learn a dictionary from input features using MiniBatch Dictionary Learning.

    Parameters
    ----------
    X : ndarray of shape [n_samples, n_features]
        Input data (e.g., phase cycle measurements for each voxel).
    n_atoms : int
        Number of dictionary atoms to learn.
    alpha : float
        Sparsity controlling parameter. Higher values enforce sparser codes.
    max_iter : int
        Maximum number of iterations for the algorithm.
    batch_size : int
        Number of samples in each mini-batch.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    dictionary : ndarray of shape [n_atoms, n_features]
        Learned dictionary atoms (each row is an atom).
    """
    # Use magnitude for real-valued dictionary learning
    X_real = np.abs(X) if np.iscomplexobj(X) else X

    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        alpha=alpha,
        max_iter=max_iter,
        batch_size=batch_size,
        random_state=random_state,
    )
    dl.fit(X_real)
    return dl.components_


def sparse_encode(
    X: npt.NDArray,
    dictionary: npt.NDArray,
    n_nonzero: int | None = None,
) -> npt.NDArray:
    """
    Encode X using sparse coding with a learned dictionary.

    Uses Orthogonal Matching Pursuit (OMP) for sparse representation.

    Parameters
    ----------
    X : ndarray of shape [n_samples, n_features]
        Input data to encode.
    dictionary : ndarray of shape [n_atoms, n_features]
        Dictionary atoms (each row is an atom).
    n_nonzero : int or None
        Maximum number of non-zero coefficients per sample.
        If None, defaults to 10% of n_atoms (minimum 1).

    Returns
    -------
    sparse_codes : ndarray of shape [n_samples, n_atoms]
        Sparse representation of X in the dictionary basis.
    """
    n_atoms = dictionary.shape[0]
    if n_nonzero is None:
        n_nonzero = max(1, n_atoms // 10)

    # Use magnitude for encoding
    X_real = np.abs(X) if np.iscomplexobj(X) else X

    # Encode each sample using OMP
    sparse_codes = np.zeros((X_real.shape[0], n_atoms), dtype=X_real.dtype)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero)
    for i, x in enumerate(X_real):
        omp.fit(dictionary.T, x)
        sparse_codes[i] = omp.coef_

    return sparse_codes


def sparse_encode_batch(
    X: npt.NDArray,
    dictionary: npt.NDArray,
    n_nonzero: int | None = None,
) -> npt.NDArray:
    """
    Batch version of sparse_encode using sklearn's batch OMP.

    Parameters
    ----------
    X : ndarray of shape [n_samples, n_features]
        Input data to encode.
    dictionary : ndarray of shape [n_atoms, n_features]
        Dictionary atoms.
    n_nonzero : int or None
        Maximum number of non-zero coefficients per sample.

    Returns
    -------
    sparse_codes : ndarray of shape [n_samples, n_atoms]
        Sparse representation of X.
    """
    from sklearn.linear_model import orthogonal_mp_gram

    n_atoms = dictionary.shape[0]
    if n_nonzero is None:
        n_nonzero = max(1, n_atoms // 10)

    X_real = np.abs(X) if np.iscomplexobj(X) else X

    # Precompute Gram matrix for efficiency
    gram = dictionary @ dictionary.T
    Xy = dictionary @ X_real.T

    sparse_codes = orthogonal_mp_gram(
        gram, Xy, n_nonzero_coefs=n_nonzero
    ).T

    return sparse_codes


def train_gasp_dict_learning(
    I: npt.NDArray,
    D: npt.NDArray,
    n_atoms: int = 16,
    alpha: float = 1.0,
    n_nonzero: int | None = None,
    useL2: bool = False,
    lam: float = 1e-2,
    max_iter: int = 1000,
    random_state: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Train GASP using dictionary learning and sparse coding.

    Pipeline:
    1. Learn dictionary from input features
    2. Encode all samples as sparse combinations of atoms
    3. Fit linear mapping from sparse codes to desired profile

    Parameters
    ----------
    I : ndarray of shape [H, W, n_features]
        Input image data (e.g., phase cycles).
    D : ndarray
        Desired 1D spectral profile.
    n_atoms : int
        Number of dictionary atoms to learn.
    alpha : float
        Sparsity penalty for dictionary learning.
    n_nonzero : int or None
        Max non-zero coefficients per sample in sparse coding.
    useL2 : bool
        Use L2 regularization for final coefficient fitting.
    lam : float
        Regularization strength if useL2=True.
    max_iter : int
        Max iterations for dictionary learning.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    output : ndarray of shape [H, W]
        Reconstructed output image.
    coefficients : ndarray of shape [n_atoms,]
        Learned mapping from sparse codes to output.
    dictionary : ndarray of shape [n_atoms, n_features]
        Learned dictionary atoms.
    """
    X, shape = _to_matrix(I)
    Dv = _repeat_profile(D, X.shape[0])

    # Step 1: Learn dictionary
    dictionary = learn_dictionary(
        X,
        n_atoms=n_atoms,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
    )

    # Step 2: Sparse encode
    sparse_codes = sparse_encode_batch(X, dictionary, n_nonzero=n_nonzero)

    # Step 3: Fit mapping from sparse codes to desired profile
    if useL2:
        # Ridge regression
        n_terms = sparse_codes.shape[1]
        L = np.eye(n_terms, dtype=sparse_codes.dtype)
        StS = sparse_codes.T @ sparse_codes
        StD = sparse_codes.T @ Dv
        coefficients = np.linalg.solve(StS + lam * L, StD)
    else:
        # Least squares
        coefficients = np.linalg.lstsq(sparse_codes, Dv, rcond=None)[0]

    # Reconstruct output
    output = (sparse_codes @ coefficients).reshape(shape)

    return output, coefficients, dictionary


def run_gasp_dict_learning(
    I: npt.NDArray,
    coefficients: npt.NDArray,
    dictionary: npt.NDArray,
    n_nonzero: int | None = None,
) -> npt.NDArray:
    """
    Apply a trained dictionary-learning GASP model to new data.

    Parameters
    ----------
    I : ndarray of shape [H, W, n_features]
        Input image data.
    coefficients : ndarray of shape [n_atoms,]
        Learned mapping from sparse codes to output.
    dictionary : ndarray of shape [n_atoms, n_features]
        Learned dictionary atoms.
    n_nonzero : int or None
        Max non-zero coefficients (should match training).

    Returns
    -------
    output : ndarray of shape [H, W]
        Output image.
    """
    X, shape = _to_matrix(I)

    # Encode using learned dictionary
    sparse_codes = sparse_encode_batch(X, dictionary, n_nonzero=n_nonzero)

    # Apply learned mapping
    output = (sparse_codes @ coefficients).reshape(shape)

    return output


def visualize_dictionary(
    dictionary: npt.NDArray,
    n_cols: int = 4,
    figsize: tuple[int, int] | None = None,
):
    """
    Visualize dictionary atoms as line plots.

    Parameters
    ----------
    dictionary : ndarray of shape [n_atoms, n_features]
        Dictionary atoms to visualize.
    n_cols : int
        Number of columns in the subplot grid.
    figsize : tuple or None
        Figure size (width, height).

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    n_atoms = dictionary.shape[0]
    n_rows = (n_atoms + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (3 * n_cols, 2 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i in range(n_atoms):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        ax.plot(dictionary[i], 'b-', linewidth=1.5)
        ax.set_title(f'Atom {i+1}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_atoms, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig, axes
