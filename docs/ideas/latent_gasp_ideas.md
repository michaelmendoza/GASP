# Latent Space Approaches for Multi-Tissue GASP

## Problem Statement

The current GASP model solves `Φ·A = D` with a single coefficient vector `A` for all voxels. This assumes the mapping from signal features to spectral profile is **spatially uniform**. However, different T2/T1 ratios create fundamentally different signal shapes in SSFP acquisitions, so a single linear/polynomial mapping struggles with heterogeneous tissues.

### Why T2/T1 Matters

The SSFP signal magnitude and phase depend on:
```
E1 = exp(-TR/T1)
E2 = exp(-TR/T2)
```

Different T2/T1 ratios produce:
- Different signal magnitudes across phase cycles
- Different spectral band shapes
- Different sensitivity to off-resonance

A GASP model trained on one T2/T1 regime may not generalize to others.

---

## Proposed Approaches

### 1. Latent-Conditioned GASP (Supervised Pre-training)

Since we can generate training data for any T2/T1 samples, we can learn a latent representation that encodes tissue properties.

**Training Phase:**
1. Simulate SSFP signals for a grid of T2/T1 ratios (e.g., 0.01 to 0.5)
2. Train an encoder network: `signal → latent z` (implicitly encodes T2/T1)
3. Train GASP coefficients conditioned on z: `A(z) = f(z)`

**Inference:**
1. For each voxel, encode `signal → z`
2. Compute voxel-specific coefficients `A(z)`
3. Apply: `output = Φ(voxel) · A(z)`

**Architecture:**
```
Signal [n_features] → Encoder → z [latent_dim] → Coefficient Network → A [n_coeffs]
                                     ↑
                      (implicitly learns T2/T1 ratio)
```

**Advantages:**
- End-to-end learnable
- Continuous adaptation to any T2/T1
- Can leverage large simulated datasets

**Disadvantages:**
- Requires neural network training
- Less interpretable than traditional GASP
- Sim-to-real gap may require fine-tuning

---

### 2. Mixture of Experts GASP

Train multiple GASP models for different T2/T1 regimes, then learn soft gating.

**Concept:**
```python
# K experts for different T2/T1 regimes
A_1, A_2, ..., A_K = train_gasp_per_regime(training_data, regimes)

# Gating network predicts weights from signal
w = softmax(gating_network(signal))  # shape: [K]

# Final output is weighted combination
output = sum(w_k * (Phi @ A_k) for k in range(K))
```

**Implementation Sketch:**
```python
class MixtureOfExpertsGASP:
    def __init__(self, n_experts=5, method='quad-cross'):
        self.n_experts = n_experts
        self.method = method
        self.expert_coefficients = []  # List of A vectors
        self.gating_network = None     # Small MLP or learned weights

    def train(self, signals, desired_profiles, t2_t1_ratios):
        """
        Train on simulated data with known T2/T1 ratios.

        signals: [n_samples, n_features]
        desired_profiles: [n_samples] or single profile
        t2_t1_ratios: [n_samples] - used to partition training
        """
        # 1. Partition training data into K regimes
        regime_edges = np.linspace(
            t2_t1_ratios.min(),
            t2_t1_ratios.max(),
            self.n_experts + 1
        )

        # 2. Train expert GASP for each regime
        for k in range(self.n_experts):
            mask = (t2_t1_ratios >= regime_edges[k]) & (t2_t1_ratios < regime_edges[k+1])
            A_k = train_gasp(signals[mask], desired_profiles[mask], method=self.method)
            self.expert_coefficients.append(A_k)

        # 3. Train gating network to predict soft expert weights from signal
        self.gating_network = self._train_gating(signals, t2_t1_ratios, regime_edges)

    def apply(self, signal):
        """Apply mixture of experts to new signal."""
        weights = self.gating_network(signal)  # [K] softmax weights
        Phi = _design_matrix(signal, self.method)

        output = sum(w * (Phi @ A) for w, A in zip(weights, self.expert_coefficients))
        return output
```

**Advantages:**
- Builds on existing GASP infrastructure
- Interpretable (each expert handles a T2/T1 range)
- Can analyze which expert activates for different tissues
- Simpler than fully neural approach

**Disadvantages:**
- Discrete regimes may not capture continuous variation perfectly
- Requires choosing number of experts

---

### 3. Variational Latent GASP (VAE-style)

Learn a continuous latent space that disentangles T2/T1 from other signal properties.

**Architecture:**
```
Encoder: signal → μ, σ → z ~ N(μ, σ)
Decoder: z → reconstructed signal
GASP Head: z → coefficients A(z)

Loss = reconstruction_loss + β * KL_divergence + gasp_fitting_loss
```

**Training Objective:**
```python
def vae_gasp_loss(signal, desired_profile, model):
    # Encode
    mu, log_var = model.encoder(signal)
    z = reparameterize(mu, log_var)

    # Decode (reconstruction)
    signal_recon = model.decoder(z)
    recon_loss = mse(signal_recon, signal)

    # KL divergence
    kl_loss = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

    # GASP fitting loss
    A = model.coefficient_head(z)
    Phi = design_matrix(signal)
    gasp_output = Phi @ A
    gasp_loss = mse(gasp_output, desired_profile)

    return recon_loss + beta * kl_loss + gasp_loss
```

**Advantages:**
- Smooth, interpolatable latent space
- Can sample and visualize latent structure
- Regularization through KL term prevents overfitting

**Disadvantages:**
- More complex training
- May require tuning β parameter
- Reconstruction may not be necessary for the task

---

### 4. Self-Supervised Contrastive Pre-training

If we want to reduce reliance on simulations and leverage unlabeled real data.

**Phase 1: Contrastive Pre-training**
```
1. Generate augmented views of signals (noise, phase perturbations)
2. Train contrastive encoder: similar T2/T1 → close in latent space
3. Use SimCLR/MoCo-style objective
```

**Phase 2: GASP Fine-tuning**
```
1. Freeze encoder (or fine-tune with small LR)
2. Train GASP coefficient head on simulated data with known T2/T1
```

**Contrastive Loss:**
```python
def contrastive_loss(z_i, z_j, temperature=0.5):
    """z_i, z_j are embeddings of augmented views of same signal."""
    similarity = cosine_similarity(z_i, z_j) / temperature
    # Positive pair should have high similarity
    # Negative pairs (other samples in batch) should have low similarity
    return cross_entropy_loss(similarity, labels)
```

**Advantages:**
- Can leverage unlabeled real data
- Pre-trained encoder may generalize better
- Reduces sim-to-real gap

**Disadvantages:**
- Two-stage training
- Augmentation strategy needs careful design

---

### 5. Spatial-Contextual Approach

Use neighboring voxels to provide context for tissue identification.

**Concept:**
```
For each voxel:
  - Extract local patch of signals [patch_size, patch_size, n_features]
  - Encode patch → latent z (captures local tissue structure)
  - Predict voxel-specific coefficients from z
```

**Architecture:**
```
Patch [P, P, n_features] → CNN Encoder → z → Coefficient MLP → A
```

**Advantages:**
- Spatial context helps disambiguate tissues
- Natural fit for CNN architectures
- May improve robustness to noise

**Disadvantages:**
- Increased computational cost
- Edge effects at tissue boundaries
- Requires spatial data structure

---

## Training Data Generation Strategy

Since we can generate data for any T2/T1 ratio:

```python
def generate_training_data(n_samples=10000, t2_t1_range=(0.01, 0.5)):
    """Generate diverse training data covering T2/T1 space."""

    # Sample T2/T1 ratios (log-uniform for better coverage of short T2)
    t2_t1_ratios = np.exp(np.random.uniform(
        np.log(t2_t1_range[0]),
        np.log(t2_t1_range[1]),
        n_samples
    ))

    # Sample T1 values (physiological range)
    t1_values = np.random.uniform(0.2, 2.0, n_samples)  # seconds
    t2_values = t2_t1_ratios * t1_values

    # Also vary off-resonance, flip angle for robustness
    f0_values = np.random.uniform(-200, 200, n_samples)  # Hz
    alpha_values = np.random.uniform(10, 90, n_samples)  # degrees

    # Generate SSFP signals using existing ssfp.py
    signals = []
    for t1, t2, f0, alpha in zip(t1_values, t2_values, f0_values, alpha_values):
        sig = ssfp_signal(t1=t1, t2=t2, f0=f0, alpha=alpha, ...)
        signals.append(sig.flatten())

    return np.array(signals), t2_t1_ratios, t1_values, t2_values
```

### Sampling Considerations

| Parameter | Range | Distribution | Rationale |
|-----------|-------|--------------|-----------|
| T2/T1 | 0.01 - 0.5 | Log-uniform | Better coverage of short T2 tissues |
| T1 | 0.2 - 2.0 s | Uniform | Physiological range |
| f0 | -200 - 200 Hz | Uniform | Off-resonance variation |
| α | 10 - 90° | Uniform | Cover flip angle range |
| SNR | 10 - 100 | Log-uniform | Robustness to noise |

---

## Recommended Implementation Path

### Phase 1: Mixture of Experts (Baseline)

Start here because it:
- Builds directly on existing GASP code
- Is interpretable and debuggable
- Provides a strong baseline for comparison

**Steps:**
1. Generate training data with diverse T2/T1 ratios
2. Partition into K=5 regimes
3. Train separate GASP models per regime
4. Implement simple gating (e.g., signal magnitude features)
5. Evaluate on heterogeneous phantom/real data

### Phase 2: Neural Gating

Replace simple gating with learned MLP:
1. Train small MLP to predict expert weights from signal
2. End-to-end fine-tuning of gating + experts
3. Compare to Phase 1 baseline

### Phase 3: Latent-Conditioned GASP

Full neural approach:
1. Implement encoder network (signal → latent)
2. Implement coefficient prediction head (latent → A)
3. Train end-to-end with GASP fitting loss
4. Add VAE regularization if needed

### Phase 4: Real Data Validation

1. Test on multi-tissue phantoms
2. Test on in-vivo data (brain, knee, etc.)
3. Compare to single-model GASP baseline
4. Analyze failure cases and iterate

---

## Evaluation Metrics

### Simulation Metrics
- **MSE** between GASP output and desired profile
- **SSIM** for structural similarity
- **Per-tissue MSE** broken down by T2/T1 regime

### Real Data Metrics
- **Water/fat separation quality** (if applicable)
- **Tissue boundary sharpness**
- **Noise amplification** (g-factor equivalent)
- **Artifact reduction** compared to single-model GASP

---

## Open Questions

1. **What T2/T1 range matters most for your application?**
   - Brain: 0.05-0.15 (white/gray matter)
   - Fluids: 0.3-0.5 (CSF, blood)
   - Musculoskeletal: 0.02-0.1 (cartilage, tendon)

2. **How many experts/regimes are sufficient?**
   - Start with 3-5, increase based on validation

3. **Should the latent also capture off-resonance?**
   - Field map variations affect signal shape too
   - May need multi-dimensional latent

4. **What's the target application?**
   - Water/fat separation?
   - Banding artifact reduction?
   - Quantitative imaging?

5. **Computational constraints?**
   - Real-time reconstruction needs?
   - GPU availability for neural approaches?

---

## References

- Mixture of Experts: Jacobs et al., "Adaptive Mixtures of Local Experts" (1991)
- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Contrastive Learning: Chen et al., "SimCLR" (2020)
- SSFP Signal Model: Scheffler & Lehnhardt, "Principles and applications of bSSFP" (2003)
