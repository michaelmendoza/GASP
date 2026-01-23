# Conditional GASP: Multi-Tissue Latent Space Implementation Plan

## Problem Statement

Current GASP uses polynomial regression to map phase-cycled bSSFP signals to spectral profiles. The bSSFP signal shape depends on T2/T1 ratio (via E1=exp(-TR/T1), E2=exp(-TR/T2)), so a model trained on one tissue type fails for heterogeneous multi-tissue data.

**Constraint**: T2/T1 values are unknown at inference - only raw signals available.

## Solution: Conditional GASP

A neural encoder learns to infer latent tissue properties from the signal, then predicts tissue-specific polynomial coefficients that work with the existing GASP design matrix.

### Architecture

```
Input Signal: x ∈ ℂ^(N_pc × N_TR)
     ↓
Tissue Encoder: z = E_θ(x) ∈ ℝ^d  (latent tissue embedding)
     ↓
Coefficient Predictor: A = C_φ(z) ∈ ℂ^k  (GASP polynomial coefficients)
     ↓
Output: y = Φ(x) @ A  (reuse existing _design_matrix)
```

### Network Design

```python
class ConditionalGASP(nn.Module):
    def __init__(self, n_acquisitions, latent_dim=16, n_coeffs=49, method='affine'):
        super().__init__()

        # Encoder: signal -> latent tissue embedding
        self.encoder = nn.Sequential(
            nn.Linear(n_acquisitions * 2, 128),  # *2 for real/imag
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )

        # Coefficient predictor: latent -> GASP coefficients
        self.coeff_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_coeffs * 2)  # *2 for complex output
        )

    def forward(self, x):
        # x: [batch, n_acquisitions] complex
        x_real_imag = torch.cat([x.real, x.imag], dim=-1)
        z = self.encoder(x_real_imag)
        A_flat = self.coeff_predictor(z)
        A = A_flat[..., :n_coeffs] + 1j * A_flat[..., n_coeffs:]
        return A, z
```

## Implementation Steps

### Step 1: Create ML Module Structure

Create new `gasp/ml/` directory with:

```
gasp/ml/
├── __init__.py
├── conditional_gasp.py   # Main model
├── data_generator.py     # Training data simulation
├── trainer.py            # Training loop
└── losses.py             # Loss functions
```

### Step 2: Training Data Generator

Leverage existing `simulate_ssfp_simple()` to generate diverse T1/T2 samples:

```python
def generate_training_batch(
    batch_size,
    acquisition_params,
    t1_range=(0.1, 4.0),
    t2_t1_ratio_range=(0.01, 0.5)
):
    """Generate training batch spanning physiological T1/T2 range."""
    signals = []
    tissue_params = []

    for _ in range(batch_size):
        T1 = np.random.uniform(*t1_range)
        ratio = np.random.uniform(*t2_t1_ratio_range)
        T2 = T1 * ratio  # Ensures T2 < T1

        signal = simulate_ssfp_simple(T1=T1, T2=T2, params=acquisition_params)
        signals.append(signal)
        tissue_params.append([T1, T2])

    return np.array(signals), np.array(tissue_params)
```

### Step 3: Loss Function

Multi-component loss for stable training:

```python
def conditional_gasp_loss(pred_output, target_profile, A_pred, z, lambda_l2=1e-3, lambda_smooth=1e-4):
    # Primary: spectral profile reconstruction
    L_profile = F.mse_loss(pred_output.abs(), target_profile)

    # Regularization: prevent coefficient explosion
    L_l2 = lambda_l2 * (A_pred.abs() ** 2).mean()

    # Smoothness: encourage similar tissues to have similar embeddings
    L_smooth = lambda_smooth * embedding_smoothness_loss(z)

    return L_profile + L_l2 + L_smooth
```

### Step 4: Training Loop

```python
def train_conditional_gasp(model, acquisition_params, target_profile, n_epochs=1000, batch_size=256):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        # Generate fresh batch each epoch (infinite data from simulation)
        signals, params = generate_training_batch(batch_size, acquisition_params)
        signals_tensor = torch.from_numpy(signals).cuda()

        # Forward pass
        A_pred, z = model(signals_tensor)
        Phi = design_matrix_torch(signals_tensor, method='affine')
        output = Phi @ A_pred

        # Loss and backprop
        loss = conditional_gasp_loss(output, target_profile, A_pred, z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### Step 5: Integration with Existing GASP API

Add wrapper functions in `gasp/gasp.py`:

```python
def train_conditional_gasp(
    I: npt.NDArray,
    D: npt.NDArray,
    method: str = "affine",
    model_path: str = None,
    **train_kwargs
) -> tuple[npt.NDArray, nn.Module]:
    """Train or apply Conditional GASP."""
    from gasp.ml.conditional_gasp import ConditionalGASP, train_model

    if model_path and os.path.exists(model_path):
        model = ConditionalGASP.load(model_path)
    else:
        model = train_model(I, D, method, **train_kwargs)

    return run_conditional_gasp(I, D, model, method), model


def run_conditional_gasp(I, D, model, method="affine"):
    """Apply trained Conditional GASP to new data."""
    X, shape = _to_matrix(I)
    A_pred, z = model(torch.from_numpy(X))
    Phi = _design_matrix(X, method)
    out = (Phi @ A_pred.numpy()).reshape(shape)
    return out
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `gasp/ml/__init__.py` | Create | Module exports |
| `gasp/ml/conditional_gasp.py` | Create | ConditionalGASP model class |
| `gasp/ml/data_generator.py` | Create | T1/T2 sampling and batch generation |
| `gasp/ml/trainer.py` | Create | Training loop and utilities |
| `gasp/ml/losses.py` | Create | Loss functions |
| `gasp/gasp.py` | Modify | Add `train_conditional_gasp`, `run_conditional_gasp` |
| `pyproject.toml` | Modify | Add torch dependency |

## Verification Plan

### 1. Unit Tests

- Test data generator produces valid T1/T2 combinations
- Test model forward pass with synthetic input
- Test coefficient prediction shapes match design matrix

### 2. Integration Tests

- Train on simulated multi-tissue phantom
- Compare reconstruction error vs standard GASP on:
  - Homogeneous tissue (should match standard GASP)
  - Heterogeneous tissue (should improve significantly)

### 3. Visualization Validation

- Plot learned latent space (t-SNE/UMAP colored by T2/T1 ratio)
- Verify tissues with similar T2/T1 cluster together
- Plot spectral profile reconstruction for different tissue types

### 4. Benchmark Metrics

```python
# Test on multi-tissue phantom
tissues = ['water', 'white-matter', 'gray-matter', 'fat', 'muscle']
for tissue in tissues:
    error_standard = compute_error(standard_gasp_output, target)
    error_conditional = compute_error(conditional_gasp_output, target)
    print(f"{tissue}: Standard={error_standard:.4f}, Conditional={error_conditional:.4f}")
```

## Dependencies to Add

```toml
[project.optional-dependencies]
ml = [
    "torch>=2.0",
    "tqdm",
]
```

## Key Design Decisions

1. **Separate encoder and coefficient predictor**: Allows inspection of learned tissue embeddings
2. **Reuse existing `_design_matrix`**: Maintains compatibility with current GASP methods
3. **Complex-valued output via real/imag split**: Avoids complex-valued neural network libraries
4. **Fresh batch each epoch**: Infinite simulated data prevents overfitting
5. **LayerNorm + GELU**: Modern architecture choices for stable training

## Alternative Approaches Considered

### Dictionary Learning (Simpler)
K discrete tissue archetypes with soft blending. Lower complexity but discrete approximation.

### Mixture of Experts (Similar complexity)
Multiple GASP experts with learned gating. Risk of expert collapse.

### VAE (More complex)
Full generative model with uncertainty quantification. Higher training complexity.

The Conditional GASP approach was chosen as a balance between performance and implementation complexity.
