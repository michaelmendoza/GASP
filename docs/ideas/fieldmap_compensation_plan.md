# Field Map Compensation for GASP

## Problem Statement

GASP matches spectra for T2/T1 and flip angle (alpha), but field map variations cause the spectral match to degrade. In the SSFP signal model, the field map enters as:

```python
beta = 2 * np.pi * (f0 + field_map) * TR
```

This means field map variations cause **voxel-specific phase shifts** that effectively "slide" each voxel's spectral profile along the frequency axis. When GASP coefficients are trained assuming a fixed field map, voxels with different off-resonance frequencies see a shifted version of the desired filter.

---

## Approach 0: Multi-Shifted Dictionary (Original Idea)

**Concept:** Generate signal models at multiple discrete field map offsets, then select/interpolate the correct one per-voxel based on the measured field map.

### Implementation Outline

```python
# 1. Generate dictionary of GASP coefficients at discrete field map offsets
field_offsets = np.linspace(-200, 200, 21)  # Hz, e.g., 21 bins
A_dict = {}  # coefficients per offset

for f_off in field_offsets:
    M_sim = ssfp(T1, T2, TR, TE, alpha, pcs, field_map=f_off)
    _, A = train_gasp(M_sim, D, method='affine')
    A_dict[f_off] = A

# 2. At inference time, use measured field map to select coefficients
def apply_gasp_with_fieldmap(data, field_map, A_dict):
    output = np.zeros(data.shape[:2])
    for i, j in np.ndindex(output.shape):
        f_local = field_map[i, j]
        # Find nearest (or interpolate between) dictionary entries
        A_local = interpolate_coefficients(A_dict, f_local)
        output[i, j] = apply_coefficients(data[i, j], A_local)
    return output
```

### Pros
- Conceptually simple
- Works with existing GASP framework
- No modification to core algorithm

### Cons
- Discrete bins may cause banding artifacts
- Per-voxel lookup is computationally slow
- Memory overhead for storing multiple coefficient sets

---

## Approach 1: Phase Demodulation (Recommended)

**Concept:** Derotate the acquired signal using the field map *before* applying GASP. This "undoes" the field-map-induced phase shift.

### Implementation Outline

```python
def demodulate_fieldmap(data, field_map, TR, TE, pcs):
    """Remove field map phase before GASP processing.

    Parameters
    ----------
    data : ndarray
        Complex SSFP data, shape [H, W, n_pcs] or [H, W, coils, n_pcs]
    field_map : ndarray
        Off-resonance frequency map in Hz, shape [H, W]
    TR : float
        Repetition time in seconds
    TE : float
        Echo time in seconds
    pcs : ndarray
        Phase cycling values in radians

    Returns
    -------
    data_corrected : ndarray
        Phase-demodulated data
    """
    data_corrected = data.copy()

    # Phase accumulated at echo time due to off-resonance
    phi_offset = 2 * np.pi * field_map * TE  # [H, W]

    # Derotate each phase cycle
    for pc_idx in range(len(pcs)):
        data_corrected[..., pc_idx] *= np.exp(-1j * phi_offset)

    return data_corrected

# Usage:
data_corrected = demodulate_fieldmap(data, field_map, TR, TE, pcs)
output, A = train_gasp(data_corrected, D, method='affine')
```

### Pros
- Single coefficient set works for all voxels
- Fast inference (no per-voxel lookup)
- Physically motivated - directly inverts field map effect
- Minimal code changes to existing pipeline

### Cons
- Requires accurate field map estimate
- Does not correct for intravoxel dephasing (T2* effects)
- Assumes field map is static during acquisition

---

## Approach 2: Field Map as Additional Feature

**Concept:** Include the field map estimate as an input feature to the GASP regression, allowing the model to learn field-map-dependent corrections.

### Implementation Outline

```python
def build_features_with_fieldmap(data, field_map):
    """Augment GASP features with field map information.

    Parameters
    ----------
    data : ndarray
        SSFP data, shape [H, W, n_pcs]
    field_map : ndarray
        Off-resonance map in Hz, shape [H, W]

    Returns
    -------
    features : ndarray
        Augmented feature matrix, shape [H*W, n_pcs + 1]
    """
    H, W = data.shape[:2]

    # Normalize field map to similar scale as signal features
    f_norm = (field_map - field_map.mean()) / (field_map.std() + 1e-8)

    # Concatenate as additional feature
    features = np.concatenate([
        data.reshape(H * W, -1),
        f_norm.reshape(H * W, 1)
    ], axis=1)

    return features

# Usage with quad-cross to learn field map interactions:
features = build_features_with_fieldmap(data, field_map)
output, A = train_gasp(features, D, method='quad-cross')
```

### Pros
- Learns compensation automatically from data
- Single forward pass at inference
- Can capture complex field-map-dependent effects

### Cons
- Increases feature dimensionality
- Requires representative training data covering field map range
- May overfit if training data is limited

---

## Approach 3: Iterative Joint Estimation

**Concept:** Alternately refine the field map estimate and GASP output until convergence.

### Implementation Outline

```python
def gasp_with_iterative_fieldmap(data, D, field_map_init, TR, TE, pcs,
                                  n_iter=5, method='affine'):
    """Iteratively refine field map and GASP output.

    Parameters
    ----------
    data : ndarray
        Raw SSFP data
    D : ndarray
        Desired spectral profile
    field_map_init : ndarray
        Initial field map estimate (can be zeros)
    n_iter : int
        Number of refinement iterations

    Returns
    -------
    output : ndarray
        Final GASP output
    field_map_est : ndarray
        Refined field map estimate
    """
    field_map_est = field_map_init.copy()

    for iteration in range(n_iter):
        # Step 1: Apply current field map correction
        data_corr = demodulate_fieldmap(data, field_map_est, TR, TE, pcs)

        # Step 2: Run GASP
        output, A = train_gasp(data_corr, D, method=method)

        # Step 3: Refine field map from phase residuals
        residual = data - forward_model(output, A, field_map_est)
        field_map_update = estimate_fieldmap_from_residual(residual, TR, TE)
        field_map_est = field_map_est + 0.5 * field_map_update  # damped update

    return output, field_map_est
```

### Pros
- Does not require pre-computed field map
- Self-consistent solution
- Can improve poor initial field map estimates

### Cons
- More complex implementation
- Convergence not guaranteed
- Slower than single-pass methods
- Requires forward model implementation

---

## Approach 4: Frequency-Shifted Training Augmentation

**Concept:** Train a single GASP model on data augmented with random field map offsets, making the learned coefficients robust to field map variations.

### Implementation Outline

```python
def train_gasp_fieldmap_robust(T1, T2, TR, TE, alpha, pcs, D,
                                field_range=(-200, 200), n_augmentations=50,
                                method='quad'):
    """Train GASP with field map augmentation for robustness.

    Parameters
    ----------
    field_range : tuple
        Range of field map offsets in Hz for augmentation
    n_augmentations : int
        Number of augmented training samples

    Returns
    -------
    A : ndarray
        Field-map-robust GASP coefficients
    """
    training_data = []

    # Generate training data with random field offsets
    for _ in range(n_augmentations):
        f_random = np.random.uniform(*field_range)  # Hz
        M_aug = ssfp(T1, T2, TR, TE, alpha, pcs, field_map=f_random)
        training_data.append(M_aug)

    # Stack and train on pooled data
    M_pooled = np.concatenate(training_data, axis=0)
    D_pooled = np.tile(D, n_augmentations)

    output, A = train_gasp(M_pooled, D_pooled, method=method)

    return A

# Apply the robust coefficients to new data
A_robust = train_gasp_fieldmap_robust(...)
output = apply_gasp(data, A_robust)
```

### Pros
- Single coefficient set for all field maps
- Implicit robustness without explicit field map
- Simple to implement

### Cons
- May sacrifice peak performance for robustness
- Averaging effect could blur spectral selectivity
- Requires careful tuning of augmentation range

---

## Recommendations

### Primary Recommendation: Phase Demodulation (Approach 1)

This is the recommended starting point because:
1. Physically principled - directly inverts the field map effect
2. Minimal code changes to existing GASP pipeline
3. Fast at inference time
4. Works well when a reasonable field map is available

### Hybrid Strategy

For best results, consider combining approaches:

1. **Demodulate first** (Approach 1) to remove bulk of field map effect
2. **Use small dictionary** (Approach 0) to handle residual errors from:
   - Field map estimation errors
   - Intravoxel dephasing
   - Non-linear effects

### When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Accurate field map available | Phase Demodulation (1) |
| No field map, want robustness | Training Augmentation (4) |
| Learning from large dataset | Field Map as Feature (2) |
| Poor initial field map | Iterative Joint Estimation (3) |
| Maximum flexibility needed | Multi-Shifted Dictionary (0) |

---

## Future Considerations

- **Intravoxel dephasing:** None of these approaches directly address T2* decay from intravoxel field gradients. May need explicit T2* correction or shorter TE.
- **Dynamic field maps:** For functional imaging, field maps may change over time. Consider temporal smoothing or per-volume estimation.
- **Integration with existing code:** The phase demodulation approach can be added as a preprocessing step in `gasp/analysis.py` or as an option in `train_gasp()`.
