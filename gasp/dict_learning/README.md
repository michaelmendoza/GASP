# Dictionary Learning / Sparse Coding for GASP

This module extends GASP with dictionary learning and sparse coding methods for spectral profile synthesis.

## Theory

### The Problem

In standard GASP, we map input features (phase cycles) to a desired spectral profile using polynomial basis functions:

```
output = Φ(X) @ A
```

Where `Φ(X)` is a design matrix of polynomial features (linear, quadratic, etc.).

### Dictionary Learning Approach

Instead of using fixed polynomial bases, **dictionary learning** discovers optimal basis functions (called "atoms") directly from the data:

```
X ≈ S @ D
```

Where:
- **X** `[n_samples, n_features]` — Input data (phase cycle measurements)
- **D** `[n_atoms, n_features]` — Learned dictionary (each row is an atom)
- **S** `[n_samples, n_atoms]` — Sparse codes (mostly zeros)

### What is an Atom?

An atom is a learned basis vector—a template pattern discovered from the data. Each input signal is represented as a weighted combination of a few atoms:

```
signal ≈ w₁·atom₁ + w₂·atom₂ + 0·atom₃ + ... + 0·atomₙ
```

The key insight is **sparsity**: most weights are zero, meaning each signal only uses 2-3 atoms.

### Mathematical Formulation

Dictionary learning solves:

```
minimize  ||X - S @ D||²  +  α||S||₁
   D, S
```

- The first term ensures good reconstruction
- The `α||S||₁` term (L1 penalty) enforces sparsity

### GASP with Dictionary Learning

The full pipeline:

1. **Learn Dictionary**: Discover atoms from training data
   ```
   D = learn_dictionary(X)
   ```

2. **Sparse Encode**: Represent each voxel with few atoms
   ```
   S = sparse_encode(X, D)  # mostly zeros
   ```

3. **Fit Mapping**: Learn coefficients from sparse codes to output
   ```
   A = fit(S, desired_profile)
   ```

4. **Inference**: Apply to new data
   ```
   S_new = sparse_encode(X_new, D)
   output = S_new @ A
   ```

## Why Use Dictionary Learning for GASP?

| Advantage | Description |
|-----------|-------------|
| **Adaptive Basis** | Atoms are learned from your data, not predefined |
| **Noise Robustness** | Sparse representation ignores noise (not captured by atoms) |
| **Interpretability** | Atoms often correspond to physical tissue patterns |
| **Efficiency** | Sparse codes compress information to few coefficients |

## Parameters

### `n_atoms` (default: 16)

Number of dictionary atoms to learn. Guidelines:
- More atoms → more expressive, but risk of overfitting
- Fewer atoms → simpler model, may underfit
- Start with `n_atoms = 2 × n_features` and tune

### `alpha` (default: 1.0)

Sparsity penalty during dictionary learning:
- Higher `alpha` → sparser codes, fewer active atoms per sample
- Lower `alpha` → denser codes, more atoms used
- Typical range: 0.1 to 10.0

### `n_nonzero` (default: None → 10% of n_atoms)

Maximum non-zero coefficients per sample during sparse coding:
- Controls sparsity during encoding
- Lower values → stricter sparsity constraint
- `None` uses 10% of atoms (minimum 1)

### `useL2` / `lam`

Optional L2 regularization for the final coefficient fitting:
- Helps when sparse codes are noisy
- `lam` controls regularization strength

## Usage

### Basic Training

```python
from gasp.dict_learning import train_gasp_dict_learning, run_gasp_dict_learning
from gasp import simulate_ssfp
from gasp.responses import gaussian

# Generate training data
M = simulate_ssfp(width=128, npcs=8)
D = gaussian(128, bw=0.2, shift=0)

# Train with dictionary learning
output, coeffs, dictionary = train_gasp_dict_learning(
    M, D,
    n_atoms=16,
    alpha=1.0,
    n_nonzero=3,
)

# Apply to new data
new_output = run_gasp_dict_learning(new_M, coeffs, dictionary, n_nonzero=3)
```

### Visualizing Atoms

```python
from gasp.dict_learning import visualize_dictionary

fig, axes = visualize_dictionary(dictionary, n_cols=4)
```

## Comparison to Standard GASP Methods

| Method | Basis Type | Sparsity | Adaptivity | Best For |
|--------|------------|----------|------------|----------|
| `linear` | Raw features | No | No | Simple, fast baseline |
| `affine` | Features + bias | No | No | Default choice |
| `quad` | Polynomial | No | No | Non-linear relationships |
| `dict-learning` | Learned atoms | **Yes** | **Yes** | Noise robustness, interpretability |

## Algorithm Details

### Dictionary Learning

Uses **MiniBatch Dictionary Learning** from scikit-learn:
- Stochastic optimization for scalability
- Processes data in mini-batches
- Atoms are L2-normalized

### Sparse Coding

Uses **Orthogonal Matching Pursuit (OMP)**:
- Greedy algorithm for sparse approximation
- Selects atoms one at a time
- Guarantees exactly `n_nonzero` active atoms

### Complex Data Handling

For complex-valued MRI data:
- Dictionary learning uses magnitude: `|X|`
- Atoms are real-valued
- Phase information is not directly captured (limitation)

## References

1. Olshausen & Field (1996). "Emergence of simple-cell receptive field properties by learning a sparse code for natural images." Nature.

2. Mairal et al. (2009). "Online dictionary learning for sparse coding." ICML.

3. Rubinstein et al. (2010). "Dictionaries for sparse representation modeling." IEEE Proceedings.
