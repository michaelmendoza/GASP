# GASP Model Overview and Evaluation Summary

## What is GASP?

**GASP** (Generation of Arbitrary Spectral Profiles) is a machine learning approach for MRI signal processing that learns to produce arbitrary spectral responses from balanced Steady-State Free Precession (bSSFP) acquisitions.

### The Core Problem

In MRI, different tissues (fat, water, muscle, etc.) have different spectral signatures based on their T1, T2, and chemical shift properties. GASP provides a data-driven way to:

1. **Synthesize arbitrary spectral profiles** from multi-phase-cycle acquisitions
2. **Separate tissues** (e.g., fat vs. water) without specialized pulse sequences
3. **Generate custom contrast** by designing target spectral responses

### How It Works

The algorithm performs **polynomial regression** to learn coefficients **A** such that:

```
D(spectral) ≈ Φ(I₁, I₂, ..., Iₙ) @ A
```

Where:
- **I** = multi-dimensional input (phase cycles, TRs, coils)
- **Φ** = design matrix (linear, affine, quadratic, or quad-cross)
- **D** = target spectral profile (Gaussian, square, sinc, etc.)

### Design Matrix Options

| Method | Design Matrix | Use Case |
|--------|---------------|----------|
| Linear | `Φ = [X]` | Simplest, raw features only |
| Affine | `Φ = [1, X]` | Adds bias term |
| Quadratic | `Φ = [1, X, X²]` | Captures nonlinear relationships |
| Quad-Cross | `Φ = [1, X, X², XᵢXⱼ]` | Full second-order polynomial |

### Key Components

- **SSFP Simulation** (`ssfp.py`) - Physics-based signal generation
- **Tissue Model** (`tissue.py`) - Realistic T1/T2/chemical shift properties
- **Spectral Responses** (`responses.py`) - Target profiles (Gaussian, square, sinc, etc.)
- **Core Algorithm** (`gasp.py`) - Regression-based coefficient learning
- **Multi-Coil Support** - Per-coil GASP with RSS combination

---

## Measuring GASP Effectiveness

### Metrics Summary

| Metric | Formula | Purpose |
|--------|---------|---------|
| **MSE/RMSE** | `mean((expected - actual)²)` | Primary accuracy metric |
| **SNR** | `mean(signal) / std(noise)` | Noise robustness |
| **PSNR** | `20·log₁₀(max) - 10·log₁₀(MSE)` | Image quality |
| **SSIM** | Structural similarity index | Perceptual quality |
| **NRMSE** | `RMSE / (max - min)` | Scale-independent error |
| **Condition Number** | `σ_max / σ_min` | System stability |
| **Spectral Fidelity** | Pearson correlation | Profile matching |

---

## Experiment Notebook Summary

### Deduplicated Methods

| Category | Method | Notebooks |
|----------|--------|-----------|
| **Sensitivity Analysis** | Alpha (flip angle) sweeps | exp15, exp21, exp31 |
| | T1/T2 ratio sensitivity | exp35, exp36 |
| | Number of phase cycles | exp29, exp38 |
| **Noise Analysis** | Monte Carlo (100+ trials) | exp43 |
| | SNR sweep (1-1000) | exp34 |
| **Method Comparison** | Linear vs affine vs quadratic | exp29 |
| | Regularization (Tikhonov, truncated SVD) | exp38 |
| **Spectral Analysis** | SVD decomposition | exp22, exp39 |
| | Frequency domain (periodogram) | exp36 |
| **Optimization** | L-BFGS-B parameter tuning | exp40 |

### Detailed Notebook Reference

| Notebook | Primary Metric | Methodology | Key Variables |
|----------|----------------|-------------|---------------|
| exp6-7 | Visual comparison | SSFP simulation | Desired profile shape |
| exp8 | MSE | Phantom simulation | Error across spatial dimensions |
| exp15 | MSE contours | Alpha sensitivity sweep | Alpha: 10-90°, 90 points |
| exp21 | MSE contours | Nonlinear sensitivity | Levenberg-Marquardt method |
| exp22 | SVD analysis | Frequency domain | Coefficient distribution |
| exp29 | RMSE | Method comparison | 5 methods, 6-48 data points |
| exp34 | SNR, PSNR, SSIM | Noise analysis | 3 filter types, noise levels |
| exp35 | MSE 2D grid | T1/T2 ratio sensitivity | Ratio: 0.5-20, alpha: 5-90° |
| exp36 | Spectral analysis | Frequency domain | Periodogram, FFT analysis |
| exp38 | Condition number, MSE | Regularization | No reg, Tikhonov, Truncated SVD |
| exp39 | SVD, Rank, MSE | Multi-ratio training | 8 tissue ratios, cross-validation |
| exp40 | MSE | Parameter optimization | L-BFGS-B optimization |
| exp43 | SNR, theoretical noise | Monte Carlo | 100 trials, SNR: 1-1000 |

---

## Missing Evaluation Approaches

### 1. Quantitative Clinical Metrics

- [ ] Fat/water separation accuracy (fat fraction error)
- [ ] Tissue classification accuracy (sensitivity/specificity)
- [ ] Comparison to ground truth Dixon methods

### 2. Robustness Testing

- [ ] **Motion artifacts** - GASP performance with motion corruption
- [ ] **B0 inhomogeneity** - systematic off-resonance effects
- [ ] **B1 inhomogeneity** - flip angle variation across FOV

### 3. Statistical Validation

- [ ] **Cross-validation** (k-fold) for coefficient generalization
- [ ] **Confidence intervals** on metrics (beyond Monte Carlo noise)
- [ ] **Statistical significance tests** comparing methods

### 4. Computational Analysis

- [ ] **Runtime benchmarks** - training and inference time
- [ ] **Memory usage** - scalability to 3D volumes

### 5. Advanced Baselines

- [ ] **Deep learning comparison** (experiment42 may be incomplete)
- [ ] **IDEAL/Dixon** direct comparison
- [ ] **Graph-cut or segmentation-based** separation methods

### 6. In-vivo Validation

- [ ] Multi-site reproducibility testing
- [ ] Clinical outcome correlation
- [ ] Diverse anatomy (beyond knee data)

### 7. Spectral Metrics

- [ ] **Bandwidth accuracy** - does achieved BW match target?
- [ ] **Transition band sharpness** - filter roll-off quality
- [ ] **Out-of-band suppression** - stopband attenuation

---

## Alternative Methods for Learning GASP Coefficients

The current GASP implementation uses polynomial regression with design matrices. Here are alternative approaches:

### 1. Neural Networks (Partially Explored in exp42)

**Current Status**: Basic MLP implemented in `experiment42-neuralnetwork-model.ipynb`

**Variants to Explore**:
- [ ] Convolutional Neural Networks (CNNs) - exploit spatial structure
- [ ] Physics-Informed Neural Networks (PINNs) - embed SSFP equations as constraints
- [ ] Transformer-based models - attention over phase cycles

### 2. Kernel Methods

| Method | Formula | Properties |
|--------|---------|------------|
| Gaussian RBF | `K(x,y) = exp(-γ‖x-y‖²)` | Infinite-dimensional feature space |
| Laplacian | `K(x,y) = exp(-γ‖x-y‖)` | Sparser, sharper features |
| Polynomial Kernel | `K(x,y) = (x·y + c)^d` | Implicit polynomial features |

### 3. Gaussian Process Regression

- Provides uncertainty quantification (confidence intervals)
- Automatic hyperparameter tuning via marginal likelihood
- O(n³) complexity - consider sparse GP for larger data

### 4. Dictionary Learning / Sparse Coding

- Learn a dictionary of basis functions
- Represent spectra as sparse combinations
- Interpretable and noise-robust

### 5. Fourier/Harmonic Basis Functions

```
Φ = [1, cos(ωx), sin(ωx), cos(2ωx), sin(2ωx), ...]
```

- Natural fit for spectral/frequency domain problems
- Orthogonal basis (better conditioning than polynomials)

### 6. Ensemble Methods

- Random Forest Regression
- Gradient Boosting (XGBoost)
- Stacking polynomial + neural network predictions

### 7. Mixture of Experts (MoE)

- Different experts for fat vs. water regions
- Conditional computation (efficiency)

### 8. Bayesian Linear Regression

- Principled uncertainty quantification
- Automatic relevance determination (ARD) for feature selection

### Comparison Summary

| Method | Complexity | Interpretability | Uncertainty | Data Efficiency |
|--------|------------|------------------|-------------|-----------------|
| Polynomial (current) | Low | High | No | High |
| Neural Network | High | Low | No* | Low |
| Kernel Ridge | Medium | Medium | No | High |
| Gaussian Process | Medium | Medium | **Yes** | High |
| Dictionary Learning | Medium | High | No | Medium |
| Fourier Basis | Low | High | No | High |
| Bayesian Linear | Low | High | **Yes** | High |

---

## Recommended Next Steps

1. **Priority 1**: Add B0/B1 inhomogeneity robustness testing (most clinically relevant)
2. **Priority 2**: Implement proper cross-validation for coefficient stability
3. **Priority 3**: Add Dixon comparison as gold-standard baseline
4. **Priority 4**: Expand in-vivo validation beyond phantom studies
5. **Priority 5**: Explore alternative coefficient learning methods (Kernel Ridge, GP)
