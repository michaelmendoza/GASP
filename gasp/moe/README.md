# Mixture of Experts GASP (MoE-GASP)

## Overview

MoE-GASP extends the standard GASP framework to handle heterogeneous tissue data with varying T2/T1 ratios. Instead of learning a single set of coefficients that must generalize across all tissues, MoE-GASP trains K specialized "expert" models, each optimized for a specific T2/T1 regime, and uses a gating network to blend their outputs.

## Motivation

The standard GASP model learns coefficients **A** that map signal features to a desired spectral profile:

```
output = Φ(x) · A
```

This works well when tissues have similar T2/T1 ratios, but fails for heterogeneous data where different tissues produce fundamentally different SSFP signal shapes. The T2/T1 ratio strongly influences the SSFP signal profile:

| T2/T1 Range | Example Tissues | Signal Characteristics |
|-------------|-----------------|------------------------|
| 0.001 - 0.02 | Proteins, tendons | Very short T2, low signal |
| 0.02 - 0.06 | Liver, muscle | Moderate decay |
| 0.06 - 0.12 | White matter, fat | Intermediate behavior |
| 0.12 - 0.25 | Gray matter | Higher signal, slower decay |
| 0.25 - 0.50 | Fluids, water | Long T2, high signal |

A single set of GASP coefficients cannot optimally handle all these regimes simultaneously.

## Mathematical Formulation

### Standard GASP

Given input signal **x** with design matrix Φ(**x**):

```
output = Φ(x) · A
```

where **A** is the learned coefficient vector.

### MoE-GASP

MoE-GASP uses K experts, each with its own coefficient vector **A_k**:

```
output = Σ_k w_k(x) · [Φ(x) · A_k]
```

where:
- **A_k** = coefficient vector for expert k (specialized for regime k)
- **w_k(x)** = gating weight for expert k given signal **x**
- Σ_k w_k(x) = 1 (weights sum to 1 via softmax)

The gating network learns which expert(s) should handle each input signal based on its characteristics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MoE-GASP Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Signal x [n_features]                                    │
│         │                                                       │
│         ├──────────────────┬────────────────────────┐          │
│         │                  │                        │          │
│         ▼                  ▼                        ▼          │
│  ┌─────────────┐    ┌─────────────┐         ┌─────────────┐   │
│  │  Expert 1   │    │  Expert 2   │   ...   │  Expert K   │   │
│  │ (low T2/T1) │    │ (mid T2/T1) │         │(high T2/T1) │   │
│  │  Φ · A_1    │    │  Φ · A_2    │         │  Φ · A_K    │   │
│  └──────┬──────┘    └──────┬──────┘         └──────┬──────┘   │
│         │                  │                        │          │
│         ▼                  ▼                        ▼          │
│       y_1                y_2                      y_K          │
│         │                  │                        │          │
│         └──────────────────┼────────────────────────┘          │
│                            │                                    │
│                            ▼                                    │
│  Input Signal x ──► ┌─────────────┐                            │
│                     │   Gating    │ ──► [w_1, w_2, ..., w_K]   │
│                     │   Network   │      (softmax weights)      │
│                     └─────────────┘                            │
│                            │                                    │
│                            ▼                                    │
│                   output = Σ w_k · y_k                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Gating Networks

Three gating network architectures are provided:

### 1. Feature-Based Gating (`FeatureGating`)

Extracts handcrafted features from the signal and uses softmax regression:

```python
features = [
    mean(|x|),           # Signal magnitude
    std(|x|),            # Magnitude variation
    |x_max| / |x_min|,   # Dynamic range
    mean(angle(x)),      # Mean phase
    std(angle(x)),       # Phase variation
    spectral_component,  # First FFT component
]
weights = softmax(W @ features + b)
```

**Pros**: Interpretable, fast, fewer parameters
**Cons**: Limited expressiveness, requires feature engineering

### 2. MLP Gating (`MLPGating`)

Uses a small neural network for more flexible decision boundaries:

```python
hidden = ReLU(W1 @ |x| + b1)
weights = softmax(W2 @ hidden + b2)
```

**Pros**: More flexible, learns features automatically
**Cons**: More parameters, may overfit with limited data

### 3. Template Gating (`TemplateGating`)

Physics-informed approach using mean signals per regime:

```python
templates = [mean_signal_for_regime_k for k in range(K)]
similarities = [cosine_sim(x, t) for t in templates]
weights = softmax(similarities / temperature)
```

**Pros**: Physics-informed, no training needed beyond computing means
**Cons**: Assumes regimes are well-separated in signal space

### 4. Top-K Gating (`TopKGating`)

Sparse gating that only activates the top-k experts:

```python
base_weights = base_gating.predict_weights(x)
sparse_weights = keep_topk(base_weights, k)
weights = normalize(sparse_weights)
```

**Pros**: Reduced computation, encourages expert specialization
**Cons**: Sharp boundaries between regimes

## Training Procedure

### Step 1: Generate Training Data

Generate diverse SSFP signals covering the T2/T1 space:

```python
from gasp.moe import generate_ssfp_training_data

signals, t2_t1_ratios, params = generate_ssfp_training_data(
    n_samples=10000,
    TR=[0.005, 0.010, 0.020],  # Multiple TRs
    alpha=np.deg2rad(30),
    npcs=16,
    t2_t1_range=(0.001, 0.5),
    add_noise_flag=True,
    noise_sigma=0.01,
    seed=42
)
```

### Step 2: Define Desired Profile

```python
from gasp.responses import gaussian

desired = gaussian(width=16, bw=0.2, shift=0)
```

### Step 3: Train MoE-GASP

```python
from gasp.moe import train_moe_gasp, MoEGASPConfig

config = MoEGASPConfig(
    n_experts=5,
    method='quad-cross',
    gating_type='mlp',
    gating_temperature=1.0,
    useL2=True,
    lam=1e-2
)

model = train_moe_gasp(signals, desired, t2_t1_ratios, config)
```

### Step 4: Apply to New Data

```python
from gasp.moe import run_moe_gasp

output, weights = run_moe_gasp(ssfp_image, model)
dominant_expert = np.argmax(weights, axis=-1)
```

## Hyperparameters

| Parameter | Range | Notes |
|-----------|-------|-------|
| `n_experts` | 3-10 | More experts = finer T2/T1 resolution |
| `method` | 'affine', 'quad', 'quad-cross' | Higher order = more expressive |
| `gating_type` | 'feature', 'mlp', 'template' | Start with 'template', try 'mlp' for flexibility |
| `gating_temperature` | 0.1-2.0 | Lower = sharper selection; higher = smoother blending |
| `lam` | 1e-4 to 1e-1 | L2 regularization strength |
| `n_samples` | 5000-50000 | More samples = better generalization |

## Analysis and Visualization

### Expert Activation Analysis

```python
from gasp.moe import analyze_expert_activation

analysis = analyze_expert_activation(ssfp_data, model, t2_t1_map)

# Visualize dominant expert (like tissue segmentation)
plt.imshow(analysis['dominant_expert'], cmap='tab10')

# Visualize gating uncertainty
plt.imshow(analysis['expert_entropy'], cmap='hot')
```

### Model Comparison

```python
from gasp.moe import compare_moe_vs_standard

comparison = compare_moe_vs_standard(
    test_data, desired,
    moe_model, standard_coeffs
)
print(f"Improvement: {comparison['improvement']:.1f}%")
```

## Expected Benefits

1. **Better fitting for heterogeneous tissues**: Each expert specializes in its T2/T1 regime
2. **Interpretable segmentation**: Dominant expert map reveals tissue-like boundaries
3. **Uncertainty quantification**: Gating entropy indicates confidence
4. **Graceful degradation**: Soft gating blends experts for intermediate tissues

## Limitations

1. **Requires simulation**: Model trained on simulated data needs domain transfer
2. **More parameters**: K experts × n_coefficients increases model size
3. **Gating errors**: Wrong expert selection can degrade performance
4. **Computational cost**: K forward passes per voxel (mitigated by vectorization)

## Mitigation Strategies

1. **Domain randomization**: Add realistic noise, B0 inhomogeneity during training
2. **Regularization**: Use L2 regularization and proper validation
3. **Soft gating**: Use temperature > 0 for smooth expert blending
4. **Stratified training**: Ensure balanced samples across T2/T1 regimes

## References

- Sherry et al., "GASP: Graph-Assembled Signal Processing for MR Spectroscopy"
- Jacobs et al., "Adaptive Mixtures of Local Experts", Neural Computation 1991
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017
