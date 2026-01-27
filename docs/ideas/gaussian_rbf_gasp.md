# Gaussian RBF Implementation for GASP

## Overview

This document outlines the implementation plan for adding **Gaussian Radial Basis Function (RBF)** support to GASP as an alternative to the current polynomial basis functions.

### Motivation

The current GASP uses polynomial design matrices (linear, affine, quadratic, quad-cross). While effective, polynomials have limitations:

| Aspect | Polynomial | Gaussian RBF |
|--------|------------|--------------|
| Feature space | Finite, explicit | Infinite-dimensional |
| Locality | Global (affects entire domain) | Localized around centers |
| Expressiveness | Limited for complex multi-tissue signals | Better for heterogeneous tissues |
| Extrapolation | Can diverge rapidly | Bounded (decays to zero) |

### Mathematical Background

**Gaussian RBF Kernel:**
```
K(x, c) = exp(-γ ||x - c||²)
```

Where:
- `x` = input signal features (shape: `[n_features]`)
- `c` = RBF center (shape: `[n_features]`)
- `γ` = bandwidth parameter (controls width of Gaussian)

**Design Matrix Construction:**
```
Φ_rbf[i, j] = exp(-γ ||X[i] - C[j]||²)
```

Where `X` has shape `[n_samples, n_features]` and `C` has shape `[n_centers, n_features]`.

---

## Implementation Plan

### 1. Add RBF Utilities to `gasp.py`

**New Functions:**

```python
def _select_rbf_centers(
    X: npt.NDArray,
    n_centers: int,
    strategy: str = "kmeans"
) -> npt.NDArray:
    """
    Select RBF center locations from training data.

    Parameters:
        X: Training data [n_samples, n_features]
        n_centers: Number of RBF centers (K)
        strategy: "kmeans", "random", or "uniform"

    Returns:
        Centers array [n_centers, n_features]
    """

def _compute_rbf_gamma(
    X: npt.NDArray,
    gamma: float | str = "auto"
) -> float:
    """
    Compute bandwidth parameter γ.

    Parameters:
        X: Training data
        gamma: float value, or "auto" for median heuristic

    Returns:
        γ value
    """

def _rbf_basis(
    X: npt.NDArray,
    centers: npt.NDArray,
    gamma: float
) -> npt.NDArray:
    """
    Compute RBF design matrix.

    Parameters:
        X: Input data [n_samples, n_features]
        centers: RBF centers [n_centers, n_features]
        gamma: Bandwidth parameter

    Returns:
        Φ_rbf [n_samples, n_centers]
    """
```

### 2. Extend `_design_matrix()` Function

**Location:** [gasp.py:30-60](gasp/gasp.py#L30-L60)

Add new method options:
- `"rbf"` - Pure RBF basis
- `"affine-rbf"` - Bias + RBF basis
- `"hybrid"` - Bias + linear + RBF (combines expressiveness)

```python
def _design_matrix(
    X: npt.NDArray,
    method: str,
    *,
    centers: npt.NDArray | None = None,
    gamma: float = 1.0
) -> npt.NDArray:
    # ... existing methods ...

    if method == "rbf":
        if centers is None:
            raise ValueError("RBF method requires centers")
        return _rbf_basis(X, centers, gamma)

    if method == "affine-rbf":
        if centers is None:
            raise ValueError("affine-rbf method requires centers")
        rbf = _rbf_basis(X, centers, gamma)
        return np.column_stack((ones, rbf))

    if method == "hybrid":
        if centers is None:
            raise ValueError("hybrid method requires centers")
        rbf = _rbf_basis(X, centers, gamma)
        return np.column_stack((ones, X, rbf))
```

### 3. Update `train_gasp()` Function

**Location:** [gasp.py:113-138](gasp/gasp.py#L113-L138)

Add RBF-specific parameters and return trained centers for inference:

```python
def train_gasp(
    I: npt.NDArray,
    D: npt.NDArray,
    method: str = "affine",
    useL2: bool = False,
    lam: float = 1e-2,
    *,
    penalise_bias: bool = False,
    # New RBF parameters
    n_centers: int = 20,
    gamma: float | str = "auto",
    center_strategy: str = "kmeans"
) -> tuple[npt.NDArray, npt.NDArray, dict | None]:
    """
    Returns: (reconstruction, coefficients, rbf_params)

    rbf_params contains {'centers': ..., 'gamma': ...} for RBF methods,
    or None for polynomial methods.
    """
```

### 4. Update `run_gasp()` Function

**Location:** [gasp.py:101-111](gasp/gasp.py#L101-L111)

Accept RBF parameters for inference:

```python
def run_gasp(
    I: npt.NDArray,
    An: npt.NDArray,
    method: str = "affine",
    *,
    rbf_params: dict | None = None
) -> npt.NDArray:
    """
    Apply GASP model with optional RBF parameters.

    rbf_params: dict with 'centers' and 'gamma' keys (required for RBF methods)
    """
```

### 5. Update Multi-Coil Function

**Location:** [gasp.py:140-180](gasp/gasp.py#L140-L180)

Propagate RBF parameters through `train_gasp_with_coils()`.

---

## Key Design Decisions

### Center Selection Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `kmeans` | K-means clustering on X | Default, adapts to data distribution |
| `random` | Random subset of training samples | Fast, good for large datasets |
| `uniform` | Uniform grid in feature space | When data is uniformly distributed |

**Recommendation:** Start with `kmeans` as default.

### Bandwidth Parameter (γ)

**Auto-selection via median heuristic:**
```python
def _compute_rbf_gamma(X, gamma="auto"):
    if gamma == "auto":
        # Median heuristic: γ = 1 / (2 * median(||x_i - x_j||²))
        from scipy.spatial.distance import pdist
        dists = pdist(X, metric='sqeuclidean')
        median_dist = np.median(dists)
        return 1.0 / (2 * median_dist) if median_dist > 0 else 1.0
    return float(gamma)
```

### Number of Centers (K)

- **Too few:** Underfitting, poor approximation
- **Too many:** Overfitting, computational cost, numerical issues

**Guidelines:**
- Start with `K = 10-50` for typical GASP data
- Use cross-validation to tune
- L2 regularization (already supported) helps prevent overfitting

---

## Implementation Steps

### Step 1: Core RBF Functions
1. Implement `_select_rbf_centers()` with kmeans strategy
2. Implement `_compute_rbf_gamma()` with auto/manual modes
3. Implement `_rbf_basis()` for design matrix computation
4. Add unit tests for each function

### Step 2: Integrate into Design Matrix
1. Add `"rbf"`, `"affine-rbf"`, `"hybrid"` methods to `_design_matrix()`
2. Update function signature to accept `centers` and `gamma`
3. Update docstring with new method descriptions

### Step 3: Update Training API
1. Modify `train_gasp()` to handle RBF parameters
2. Return `rbf_params` dict for RBF methods
3. Ensure backward compatibility (polynomial methods unchanged)

### Step 4: Update Inference API
1. Modify `run_gasp()` to accept `rbf_params`
2. Validate that centers/gamma are provided for RBF methods

### Step 5: Multi-Coil Support
1. Update `train_gasp_with_coils()` to use shared centers across coils
2. Return combined `rbf_params`

### Step 6: Testing & Validation
1. Add tests comparing RBF vs polynomial on synthetic data
2. Test on existing phantom simulations
3. Compare MSE, conditioning, and spectral fidelity

---

## Example Usage

```python
from gasp import train_gasp, run_gasp

# Training with RBF
I_train = simulate_ssfp(...)  # [H, W, PCs, TRs]
D = gaussian(npcs=16, center=0, sigma=0.2)

# Train with Gaussian RBF
reconstruction, coeffs, rbf_params = train_gasp(
    I_train, D,
    method="affine-rbf",
    n_centers=30,
    gamma="auto",
    useL2=True,
    lam=1e-3
)

# Inference on new data
I_test = acquire_data(...)
output = run_gasp(I_test, coeffs, method="affine-rbf", rbf_params=rbf_params)
```

---

## Dependencies

**Required (already in project):**
- `numpy` - array operations
- `scipy` - optional for pdist in gamma computation

**Optional:**
- `scikit-learn` - for KMeans clustering (or implement simple version)

---

## Verification Plan

1. **Unit Tests:**
   - `test_rbf_basis_shape()` - verify output dimensions
   - `test_rbf_gamma_auto()` - verify median heuristic
   - `test_rbf_centers_kmeans()` - verify center selection

2. **Integration Tests:**
   - Compare RBF vs `quad-cross` on multi-tissue phantom
   - Verify backward compatibility with existing polynomial methods

3. **Performance Tests:**
   - MSE comparison across methods
   - Conditioning number analysis
   - Runtime benchmarks

---

## Future Extensions

1. **Multiple kernel support:** Laplacian, polynomial kernel, etc.
2. **Learned centers:** Gradient-based optimization of center locations
3. **Sparse RBF:** Use only nearby centers per sample (efficiency)
4. **Kernel combination:** Weighted sum of multiple kernels

---

## Files to Modify

| File | Changes |
|------|---------|
| [gasp/gasp.py](gasp/gasp.py) | Core implementation |
| [test.py](test.py) | Add RBF examples |
| [docs/ideas/gasp_evaluation_summary.md](docs/ideas/gasp_evaluation_summary.md) | Update with RBF method |

---

## References

- Buhmann, M. D. (2003). Radial Basis Functions: Theory and Implementations
- Scholkopf, B., & Smola, A. J. (2002). Learning with Kernels
- scikit-learn RBF kernel documentation
