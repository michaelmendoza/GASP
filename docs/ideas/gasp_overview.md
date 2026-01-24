# GASP Model Overview

**GASP (Generation of Arbitrary Spectral Profiles)** is a Python library for MRI signal simulation and spectral shaping, specifically designed for balanced Steady-State Free Precession (bSSFP) sequences.

## What GASP Does

GASP enables control over how MRI signals respond at different frequencies in bSSFP sequences. The key innovation is using a **regression-based approach** to train coefficients that map multi-dimensional MRI measurement data (across multiple phase cycles and TRs) to a desired spectral profile.

### Core Workflow

1. Simulate or acquire bSSFP data with multiple phase cycles/TRs
2. Define a desired spectral response (e.g., Gaussian, bandpass, notch filter)
3. Train GASP coefficients to map input data → desired profile
4. Apply trained model to reconstruct images with the target spectral characteristics

### Design Matrix Methods

GASP supports 4 design matrix methods with increasing complexity:

| Method | Design Matrix | Description |
|--------|---------------|-------------|
| `linear` | Φ = [X] | Basic linear mapping |
| `affine` | Φ = [1, X] | Linear with bias term |
| `quad` | Φ = [1, X, X²] | Quadratic terms |
| `quad-cross` | Φ = [1, X, X², XᵢXⱼ] | Quadratic + cross-terms |

## Main Components

| Module | Purpose |
|--------|---------|
| `gasp/gasp.py` | Core regression algorithm (`train_gasp`, `run_gasp`) |
| `gasp/simulation.py` | bSSFP sequence simulation |
| `gasp/ssfp.py` | Physics-based signal equations |
| `gasp/responses.py` | Predefined spectral profiles (Gaussian, Butterworth, Sinc, etc.) |
| `gasp/phantom.py` | Synthetic MRI phantom generation |
| `gasp/tissue.py` | Tissue property definitions (T1, T2, chemical shift) |
| `gasp/analysis.py` | Evaluation and comparison tools |
| `gasp/dataset.py` | Real MRI data loading (12+ datasets) |
| `gasp/coil.py` | Multi-coil data combination |

## Basic Usage

```python
from gasp import simulation, responses
import numpy as np

# Simulate bSSFP data
M = simulation.simulate_ssfp(
    width=256,
    height=256,
    npcs=16,
    TRs=[5e-3, 10e-3, 20e-3],
    alpha=np.deg2rad(60)
)

# Define desired spectral profile
D = responses.gaussian(256, bw=0.2, shift=0)

# Train and visualize
Ic, M, An = simulation.simulate_gasp(
    D, 256, 256, 16,
    [5e-3, 10e-3, 20e-3],
    np.deg2rad(60),
    2 * np.pi
)
simulation.view_gasp_results(Ic, M, D)
```

## Multi-Coil Processing

```python
from gasp.gasp import train_gasp_with_coils

# Real data with multiple coils: [H, W, coils, PCs, TR]
rss, An = train_gasp_with_coils(
    data, D,
    method='affine',
    useL2=True,
    lam=1e-2
)
```

---

# Measuring GASP Effectiveness

## 1. Visual Profile Comparison

Compare reconstructed spectral profiles against desired profiles:

```python
from gasp import simulation

# View results with profile overlay
simulation.view_gasp_results(Ic, M, D)

# Multi-image comparison grid
simulation.view_gasp_comparison(results_list)
```

## 2. Parameter Sweeps

Evaluate sensitivity across parameter ranges:

```python
from gasp.analysis import gasp_sweep, plot_gasp_sweep

# Sweep over bandwidth
results = gasp_sweep(
    data,
    sweep_type='bw',
    sweep_start=0.1,
    sweep_end=0.5,
    sweep_size=10
)
plot_gasp_sweep(results)

# Available sweep types: 'alpha', 'bw', 'shift'
```

## 3. Method Comparison

Compare the 4 design matrix methods:

```python
from gasp.gasp import train_gasp

methods = ['linear', 'affine', 'quad', 'quad-cross']
results = {}

for method in methods:
    Ic, An = train_gasp(data, D, method=method)
    results[method] = Ic
```

## 4. Regularization Analysis

Evaluate impact of L2 regularization:

```python
from gasp.gasp import train_gasp

lambdas = [1e-4, 1e-3, 1e-2, 1e-1]
for lam in lambdas:
    Ic, An = train_gasp(data, D, method='affine', useL2=True, lam=lam)
```

## 5. Quantitative Metrics

### Built-in: Least-Squares Error

Training minimizes:
- **Without regularization**: ||Φ(I) - D||²
- **With L2 regularization**: ||Φ(I) - D||² + λ||A||²

### Custom Metrics

```python
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_metrics(reconstructed, desired):
    """Compute evaluation metrics for GASP reconstruction."""

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((reconstructed - desired) ** 2))

    # Normalized RMSE
    nrmse = rmse / (desired.max() - desired.min())

    # Correlation coefficient
    correlation = np.corrcoef(reconstructed.flatten(), desired.flatten())[0, 1]

    # Peak Signal-to-Noise Ratio
    mse = np.mean((reconstructed - desired) ** 2)
    psnr = 10 * np.log10(desired.max() ** 2 / mse) if mse > 0 else float('inf')

    return {
        'rmse': rmse,
        'nrmse': nrmse,
        'correlation': correlation,
        'psnr': psnr
    }
```

### Structural Similarity (SSIM)

```python
from skimage.metrics import structural_similarity as ssim

# For 2D image comparison
ssim_value = ssim(reconstructed, desired, data_range=desired.max() - desired.min())
```

## 6. Real Data Validation

Compare against established methods using in-vivo datasets:

```python
from gasp.dataset import load_dataset0, load_dataset1
from gasp.analysis import gasp_train_and_run

# Load real data (knee, ankle, brain available)
data = load_dataset0()

# Compare GASP vs Dixon methods
gasp_result = gasp_train_and_run(data, D)
# Compare with 2-point or 3-point Dixon from analysis module
```

## 7. Comprehensive Evaluation Pipeline

```python
import numpy as np
from gasp import simulation, responses
from gasp.gasp import train_gasp

def evaluate_gasp(npcs_list, methods, bw_list):
    """Run comprehensive GASP evaluation."""

    results = []

    for npcs in npcs_list:
        for method in methods:
            for bw in bw_list:
                # Simulate data
                M = simulation.simulate_ssfp(
                    width=256, height=256,
                    npcs=npcs, TRs=[5e-3, 10e-3, 20e-3],
                    alpha=np.deg2rad(60)
                )

                # Define profile
                D = responses.gaussian(256, bw=bw, shift=0)

                # Train
                Ic, An = train_gasp(M, D, method=method)

                # Compute metrics
                metrics = compute_metrics(Ic, D)
                metrics.update({
                    'npcs': npcs,
                    'method': method,
                    'bw': bw
                })
                results.append(metrics)

    return results

# Run evaluation
results = evaluate_gasp(
    npcs_list=[4, 8, 16],
    methods=['linear', 'affine', 'quad', 'quad-cross'],
    bw_list=[0.1, 0.2, 0.3]
)
```

## Summary of Evaluation Approaches

| Approach | What It Measures | When to Use |
|----------|------------------|-------------|
| Visual comparison | Qualitative profile match | Initial debugging, presentations |
| Parameter sweeps | Sensitivity to parameters | Robustness analysis |
| Method comparison | Algorithm performance | Selecting best method |
| RMSE/NRMSE | Reconstruction accuracy | Quantitative benchmarking |
| Correlation | Profile shape similarity | Shape-focused evaluation |
| PSNR | Signal quality | Noise sensitivity analysis |
| SSIM | Perceptual similarity | Image quality assessment |
| Real data validation | Clinical applicability | Translation to practice |
