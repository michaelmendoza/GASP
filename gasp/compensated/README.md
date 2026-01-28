# Amplitude Compensation for Universal GASP Coefficients

This module provides T2/T1-based amplitude compensation to enable GASP coefficients
that generalize across different tissue types.

## The Problem

Standard GASP coefficients are trained on signals from a specific tissue type (with a
particular T2/T1 ratio). When applied to tissues with different T2/T1 ratios, the
reconstruction quality degrades because:

1. **bSSFP signal amplitude depends on T2/T1** - tissues with different relaxation
   properties produce different signal magnitudes
2. **The spectral shape also varies** - but GASP's polynomial design matrix can
   partially adapt to shape changes
3. **Amplitude changes are not captured** - the learned coefficients assume a
   particular signal scaling

## Theory

### bSSFP Signal Equation

The balanced steady-state free precession (bSSFP) signal is given by:

$$M_{xy} = M_0 \frac{\sin\alpha (1 - E_1)}{1 - E_1 E_2 - (E_1 - E_2)\cos\alpha} \cdot \frac{1 - E_2 \cos\theta}{1 - E_2 \cos\theta + \text{(phase terms)}}$$

where:
- $E_1 = e^{-TR/T_1}$ (longitudinal relaxation)
- $E_2 = e^{-TR/T_2}$ (transverse relaxation)
- $\alpha$ = flip angle
- $\theta = 2\pi f_{off} \cdot TR - \Delta\phi$ (phase per TR)
- $M_0$ = proton density

### On-Resonance Amplitude

At on-resonance ($\theta = 0$), the signal magnitude simplifies to:

$$A = \frac{\sin\alpha (1 - E_1)}{1 - E_1 E_2 - (E_1 - E_2)\cos\alpha}$$

This amplitude **depends strongly on T2/T1**:
- Short T2 (small $E_2$) → lower amplitude
- Long T2 (large $E_2$) → higher amplitude

### Amplitude Compensation

Given T1 and T2 values, we normalize signals by dividing out the T2-dependent amplitude:

$$S_{normalized} = \frac{S_{measured}}{A(T_2, T_1, TR, \alpha)} \cdot A(T_{2,ref}, T_{1,ref}, TR, \alpha)$$

This maps all signals to what they would be at a **reference T2/T1 ratio** (default: 0.1).

## Usage

### Primary Use Case: Known T1/T2 (Simulations)

For simulations where T1/T2 are known, use the `known_t1` and `known_t2` parameters:

```python
from gasp.compensated import train_compensated_gasp, run_compensated_gasp

TRs = [5e-3, 10e-3, 15e-3]
alpha = np.deg2rad(60)
npcs = 8

# Training on Gray Matter (T1=900ms, T2=100ms)
TRAIN_T1, TRAIN_T2 = 0.9, 0.1

_, coefficients, metadata = train_compensated_gasp(
    I_train, D, TRs, alpha, npcs,
    method='affine',
    known_t1=TRAIN_T1,  # Use known T1
    known_t2=TRAIN_T2,  # Use known T2
    useL2=True, lam=1e-2
)

# Apply to different tissue (Muscle: T1=900ms, T2=50ms)
TEST_T1, TEST_T2 = 0.9, 0.05

output = run_compensated_gasp(
    I_test, coefficients, metadata,
    method='affine',
    known_t1=TEST_T1,
    known_t2=TEST_T2
)
```

### Results with Known T1/T2

| Tissue | T2/T1 | Standard RMSE | Compensated RMSE | Improvement |
|--------|-------|---------------|------------------|-------------|
| Gray Matter (train) | 0.111 | 0.029 | 0.029 | 0% |
| Muscle | 0.056 | 0.148 | 0.041 | **+72%** |
| White Matter | 0.133 | 0.054 | 0.029 | **+46%** |
| Fat | 0.280 | 0.304 | 0.031 | **+90%** |

### T1/T2 Estimation for Real Data

For real imaging data where T1/T2 are unknown, use the PLANET algorithm with
**off-resonance** signals:

```python
from gasp.compensated import estimate_t1_t2_planet

# PLANET requires off-resonance signals (f0 != 0)
# The ellipse fitting fails for on-resonance data
t1_map, t2_map, f0_map = estimate_t1_t2_planet(
    signals,  # [H, W, npcs] - single TR, phase-cycled
    TR=5e-3,
    alpha=np.deg2rad(60),
    npcs=8
)
```

**Note**: The TR-ratio-based T2 estimation (`estimate_t2_from_tr_decay`) has limited
accuracy because bSSFP signal ratios between TRs vary by only 0.5-2% across
physiological T2 values (50-200ms). PLANET is more reliable for real data.

## Algorithm

```
Input: Multi-TR phase-cycled signals I[H, W, npcs, nTRs]
       TRs = [TR1, TR2, TR3]  (e.g., 5ms, 10ms, 15ms)
       Flip angle α
       Reference T2/T1 ratio (default: 0.1)
       known_t1, known_t2 (optional, for simulations)

1. GET T1/T2 VALUES:
   If known_t1 and known_t2 provided:
     Use directly
   Else:
     Estimate using PLANET or dictionary matching

2. COMPUTE NORMALIZATION:
   For each TR:
     A_actual = ssfp_amplitude(T1, T2, TR, α)
     A_target = ssfp_amplitude(T1_ref, T2_ref, TR, α)
     scale = A_target / A_actual

3. NORMALIZE:
   I_normalized = I * scale  (per-TR scaling)

4. APPLY GASP:
   Train or run GASP on I_normalized
```

## Assumptions and Limitations

1. **Known T1/T2 is best** - When possible (simulations, known tissue types),
   provide `known_t1` and `known_t2` for accurate compensation.

2. **PLANET requires off-resonance** - The ellipse fitting used by PLANET fails
   for on-resonance signals (they are collinear, not elliptical).

3. **TR-ratio estimation is limited** - Signal ratios between TRs vary by only
   0.5-2% across physiological T2 values, making estimation unreliable.

4. **On-resonance approximation** - The amplitude formula assumes on-resonance signal.
   Off-resonance voxels (B0 inhomogeneity) will have different amplitude behavior.

5. **SNR requirements** - T2/T1 estimation requires adequate SNR. Low-signal
   regions may have unreliable estimates.

## API Reference

### `train_compensated_gasp`
```python
train_compensated_gasp(
    I, D, TRs, alpha, npcs,
    method='affine',
    reference_t2_t1=0.1,
    known_t1=None,      # Provide for simulations
    known_t2=None,      # Provide for simulations
    estimate_t1=False,
    **gasp_kwargs
) -> (reconstruction, coefficients, metadata)
```

### `run_compensated_gasp`
```python
run_compensated_gasp(
    I, coefficients, metadata,
    method='affine',
    known_t1=None,      # Provide for simulations
    known_t2=None       # Provide for simulations
) -> output
```

### `estimate_t1_t2_planet`
```python
estimate_t1_t2_planet(
    signals,  # [H, W, npcs]
    TR, alpha, npcs
) -> (t1_map, t2_map, f0_map)
```

## References

1. Scheffler, K., & Lehnhardt, S. (2003). Principles and applications of balanced
   SSFP techniques. European Radiology, 13(11), 2409-2418.

2. Hargreaves, B. A. (2012). Rapid gradient-echo imaging. Journal of Magnetic
   Resonance Imaging, 36(6), 1300-1313.

3. Shcherbakova, Y., et al. (2018). PLANET: An ellipse fitting approach for
   simultaneous T1 and T2 mapping using phase-cycled balanced steady-state
   free precession. Magnetic Resonance in Medicine, 79(2), 711-722.
