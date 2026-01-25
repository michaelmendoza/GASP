# Conditional GASP: Adaptive Spectral Profile Generation for Multi-Tissue bSSFP MRI

## Abstract

Standard GASP (Generation of Arbitrary Spectral Profiles) uses polynomial regression to map phase-cycled bSSFP acquisitions to desired spectral profiles. However, the bSSFP signal manifold geometry varies with tissue-dependent relaxation parameters, causing single-coefficient models to fail on heterogeneous multi-tissue data. We propose Conditional GASP, a neural network approach that learns to predict tissue-adaptive polynomial coefficients by inferring latent tissue properties directly from the acquired signals.

## 1. Background

### 1.1 bSSFP Signal Model

The balanced steady-state free precession (bSSFP) signal for a voxel with relaxation times $T_1$ and $T_2$ is given by the Ernst-Anderson equation:

$$M_{xy} = M_0 \frac{(1 - E_1) \sin\alpha}{1 - E_1 E_2 - (E_1 - E_2)\cos\alpha} \cdot f(\theta)$$

where:
- $E_1 = e^{-T_R/T_1}$ and $E_2 = e^{-T_R/T_2}$ are the longitudinal and transverse relaxation factors
- $\alpha$ is the flip angle
- $T_R$ is the repetition time
- $\theta = 2\pi \Delta f \cdot T_R - \Delta\phi$ is the phase per repetition
- $\Delta f$ is the off-resonance frequency
- $\Delta\phi$ is the RF phase increment (phase cycling)

The signal magnitude exhibits characteristic banding artifacts at frequencies where $\theta = (2k+1)\pi$, with the band spacing and shape determined by the ratio $T_2/T_1$.

### 1.2 Standard GASP Formulation

GASP constructs a spectral profile by linear combination of multi-acquisition bSSFP data. Given $N$ acquisitions (phase cycles × TRs), the signal at each spatial location forms a feature vector $\mathbf{x} \in \mathbb{C}^N$.

The design matrix $\Phi$ transforms the raw signal into polynomial features:

| Method | Design Matrix $\Phi(\mathbf{x})$ |
|--------|----------------------------------|
| Linear | $[\mathbf{x}]$ |
| Affine | $[1, \mathbf{x}]$ |
| Quadratic | $[1, \mathbf{x}, \mathbf{x}^2]$ |
| Quad-Cross | $[1, \mathbf{x}, \mathbf{x}^2, \{x_i x_j\}_{i<j}]$ |

The GASP output is computed as:

$$y = \Phi(\mathbf{x}) \cdot \mathbf{A}$$

where $\mathbf{A} \in \mathbb{C}^K$ are the polynomial coefficients (K depends on method), typically found via least-squares regression:

$$\mathbf{A}^* = \arg\min_{\mathbf{A}} \|\Phi(\mathbf{X}) \mathbf{A} - \mathbf{D}\|^2 + \lambda\|\mathbf{A}\|^2$$

where $\mathbf{D}$ is the desired spectral profile and $\lambda$ is an optional L2 regularization parameter.

### 1.3 The Multi-Tissue Problem

The fundamental limitation of standard GASP is that optimal coefficients $\mathbf{A}^*$ depend on tissue relaxation properties:

$$\mathbf{A}^*(T_1, T_2) \neq \mathbf{A}^*(T_1', T_2') \quad \text{when} \quad \frac{T_2}{T_1} \neq \frac{T_2'}{T_1'}$$

This occurs because the bSSFP signal shape—specifically the band profile, null positions, and amplitude modulation—varies nonlinearly with $T_2/T_1$. Coefficients optimized for one tissue type (e.g., gray matter with $T_2/T_1 \approx 0.11$) produce suboptimal spectral profiles when applied to tissues with different ratios (e.g., CSF with $T_2/T_1 \approx 0.5$).

## 2. Conditional GASP

### 2.1 Key Insight

While $T_1$ and $T_2$ are not directly observable from a single bSSFP acquisition, the multi-acquisition signal vector $\mathbf{x}$ implicitly encodes information about the underlying tissue properties. Conditional GASP exploits this by learning to:

1. **Infer** a latent tissue representation $\mathbf{z}$ from the signal $\mathbf{x}$
2. **Predict** tissue-adaptive coefficients $\mathbf{A}(\mathbf{z})$

### 2.2 Architecture

The Conditional GASP model consists of two neural network components:

**Tissue Encoder** $E_\theta$: Maps the complex signal to a real-valued latent embedding:

$$\mathbf{z} = E_\theta(\mathbf{x}) \in \mathbb{R}^d$$

where $d$ is the latent dimension (typically 8-16). Since standard neural network layers operate on real values, complex inputs are converted via concatenation:

$$\tilde{\mathbf{x}} = [\text{Re}(\mathbf{x}), \text{Im}(\mathbf{x})] \in \mathbb{R}^{2N}$$

The encoder architecture uses fully-connected layers with LayerNorm and GELU activations:

$$E_\theta: \mathbb{R}^{2N} \xrightarrow{\text{FC}} \mathbb{R}^{128} \xrightarrow{\text{LN, GELU}} \mathbb{R}^{64} \xrightarrow{\text{LN, GELU}} \mathbb{R}^{d}$$

**Coefficient Predictor** $C_\phi$: Maps the latent embedding to complex polynomial coefficients:

$$\mathbf{A} = C_\phi(\mathbf{z}) \in \mathbb{C}^K$$

The predictor outputs $2K$ real values (real and imaginary parts):

$$C_\phi: \mathbb{R}^{d} \xrightarrow{\text{FC, GELU}} \mathbb{R}^{64} \xrightarrow{\text{FC}} \mathbb{R}^{2K}$$

$$\mathbf{A} = \mathbf{A}_{\text{real}} + i \cdot \mathbf{A}_{\text{imag}}$$

### 2.3 Forward Pass

Given input signal $\mathbf{x}$:

1. Encode tissue properties: $\mathbf{z} = E_\theta(\mathbf{x})$
2. Predict coefficients: $\mathbf{A} = C_\phi(\mathbf{z})$
3. Build design matrix: $\Phi = \Phi(\mathbf{x})$ (same as standard GASP)
4. Compute output: $y = \Phi \cdot \mathbf{A}$

Critically, the design matrix construction is identical to standard GASP—only the coefficient prediction is learned. This preserves interpretability and allows direct comparison.

### 2.4 Loss Function

The training objective combines multiple terms:

$$\mathcal{L} = \mathcal{L}_{\text{profile}} + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{smooth}}$$

**Profile Reconstruction Loss**: Mean squared error between predicted and target spectral profiles:

$$\mathcal{L}_{\text{profile}} = \frac{1}{W} \sum_{w=1}^{W} \left( |y_w| - D_w \right)^2$$

where $W$ is the spectral width and $D_w$ is the desired profile at frequency index $w$.

**Coefficient Regularization**: L2 penalty to prevent coefficient explosion:

$$\mathcal{L}_{\text{reg}} = \frac{1}{K} \sum_{k=1}^{K} |A_k|^2$$

**Embedding Smoothness** (optional): Encourages similar $T_2/T_1$ ratios to produce similar embeddings:

$$\mathcal{L}_{\text{smooth}} = -\frac{1}{B^2} \sum_{i,j} \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|}$$

This self-supervised term promotes a smooth latent manifold without requiring explicit tissue labels.

### 2.5 Training Strategy

**Data Generation**: Training data is generated via simulation using the bSSFP signal equation across the physiological parameter space:

- $T_1 \sim \text{Uniform}(0.1, 4.0)$ seconds
- $T_2/T_1 \sim \text{Uniform}(0.01, 0.5)$
- Optional Gaussian noise: $\sigma \approx 0.005$

This covers the range from short-$T_2$ tissues (tendon: $T_2/T_1 \approx 0.01$) to long-$T_2$ fluids (CSF: $T_2/T_1 \approx 0.5$).

**Optimization**: AdamW optimizer with cosine annealing learning rate schedule:

- Initial learning rate: $10^{-3}$
- Weight decay: $10^{-4}$
- Batch size: 32 samples × W frequency points
- Epochs: 50-100

**Key advantage**: Since training data is simulated, we have access to unlimited diverse samples. Fresh batches are generated each epoch, preventing overfitting.

### 2.6 Latent Representation and Continuous Coefficient Prediction

A key conceptual distinction of Conditional GASP is that it does **not** select from or interpolate between a discrete set of pre-computed coefficient vectors. Instead, it learns a **continuous mapping** from a compressed tissue representation to coefficients.

#### The Latent Space as Tissue Encoding

The encoder network compresses the high-dimensional input signal $\mathbf{x} \in \mathbb{C}^N$ into a low-dimensional latent vector $\mathbf{z} \in \mathbb{R}^d$ (typically $d=16$). This latent space serves as a learned, continuous representation of tissue-relevant signal properties.

Each dimension of $\mathbf{z}$ captures some learned feature of the input signal—these features are not explicitly specified but emerge during training. Empirically, the network learns to encode properties related to:

- Band profile shape (influenced by $T_2/T_1$ ratio)
- Signal amplitude characteristics
- Phase cycling response patterns

Signals from tissues with similar relaxation properties will map to nearby points in this latent space, while dissimilar tissues map to distant points.

#### Continuous Coefficient Prediction

The coefficient predictor implements a smooth, continuous function:

$$f_\phi: \mathbb{R}^d \rightarrow \mathbb{C}^K$$

This function maps **any point** in the $d$-dimensional latent space to a corresponding set of $K$ polynomial coefficients. Because neural networks with smooth activations (GELU) implement continuous functions, small changes in $\mathbf{z}$ produce small changes in the predicted coefficients $\mathbf{A}$.

The network learns this mapping by observing many simulated tissue examples during training, spanning the full $T_1$, $T_2/T_1$ parameter space. After training, it can output sensible coefficients for any tissue—including parameter combinations it never explicitly encountered—because the learned function is smooth and continuous. This is the key to generalization: the model does not memorize coefficient sets for specific tissues, but learns the underlying relationship between tissue properties and optimal coefficients.

This is fundamentally different from:

| Approach | Description | Limitation |
|----------|-------------|------------|
| Single coefficient set | One $\mathbf{A}$ for all voxels | Cannot adapt to tissue variation |
| Discrete selection | Choose from $M$ pre-computed sets | Limited to $M$ tissue types |
| Interpolation | Blend between $M$ prototype sets | Assumes linear coefficient space |
| **Continuous prediction** | Learn $f: \mathbf{z} \rightarrow \mathbf{A}$ | Arbitrary tissue-adaptive coefficients |

#### Effective Dimensionality

While the latent dimension $d$ determines the capacity of the tissue encoding, it does **not** limit the model to $d$ distinct coefficient sets. The coefficient predictor can generate an effectively infinite variety of coefficient vectors—one unique set for every point in the continuous $d$-dimensional latent manifold.

For a typical image with $256 \times 256$ voxels, the model predicts 65,536 potentially distinct coefficient vectors, each tailored to the inferred tissue properties of that voxel.

#### Why Compress to a Latent Space?

The latent bottleneck serves several purposes:

1. **Regularization**: Forces the network to extract only the most relevant tissue information, discarding noise and acquisition-specific variation

2. **Generalization**: A smooth, low-dimensional manifold generalizes better to unseen tissue combinations than direct signal-to-coefficient mapping

3. **Interpretability**: The latent space can be visualized and analyzed to understand what tissue properties the model has learned to distinguish

## 3. Theoretical Analysis

### 3.1 Expressiveness

The Conditional GASP architecture can represent any tissue-dependent coefficient mapping $\mathbf{A}(T_1, T_2)$ that varies smoothly with relaxation parameters. The universal approximation theorem guarantees that sufficiently wide networks can approximate this mapping to arbitrary precision.

### 3.2 Identifiability

A natural question is whether $T_2/T_1$ can be uniquely determined from the signal $\mathbf{x}$. While the exact values $(T_1, T_2)$ are not identifiable from magnitude-only data without additional constraints, the *ratio* $T_2/T_1$ is encoded in the signal shape:

- **Band width**: Wider bands indicate larger $T_2/T_1$
- **Band amplitude**: The peak-to-null ratio depends on $E_2/E_1$
- **Phase cycling response**: Different ratios produce distinct patterns across phase cycles

The encoder learns to extract these features into the latent representation $\mathbf{z}$.

### 3.3 Generalization

The model generalizes to unseen $(T_1, T_2)$ combinations because:

1. Training spans the full physiological parameter space
2. The latent space learns a continuous representation
3. Coefficient prediction is a smooth function of $\mathbf{z}$

## 4. Comparison with Standard GASP

| Aspect | Standard GASP | Conditional GASP |
|--------|---------------|------------------|
| Coefficients | Single set, fixed | Per-voxel, adaptive |
| Training data | Calibration region | Simulated across $T_1$, $T_2$ |
| Multi-tissue | Poor (single optimum) | Good (tissue-adaptive) |
| Inference | Matrix multiply | Neural network + matrix multiply |
| Interpretability | Direct (polynomial) | Indirect (via latent space) |
| Computational cost | Low | Moderate |

## 5. Implementation Notes

### 5.1 Complex Number Handling

PyTorch linear layers require real-valued inputs. Complex signals are handled by:

1. **Input**: Concatenate real and imaginary parts: $[\text{Re}(\mathbf{x}), \text{Im}(\mathbf{x})]$
2. **Output**: Split predictions into real/imaginary and reconstruct: $\mathbf{A} = \mathbf{A}_R + i\mathbf{A}_I$

### 5.2 Numerical Stability

- LayerNorm after each hidden layer stabilizes training
- GELU activation (vs. ReLU) provides smoother gradients
- L2 coefficient regularization prevents unbounded growth

### 5.3 Hyperparameter Selection

| Parameter | Recommended Value | Notes |
|-----------|-------------------|-------|
| Latent dim | 16 | 8-32 works well |
| Hidden dims | (128, 64) | Larger for quad-cross |
| $\lambda_{\text{reg}}$ | $10^{-3}$ | Increase if coefficients explode |
| $\lambda_{\text{smooth}}$ | $10^{-4}$ | Optional, helps interpretability |
| Learning rate | $10^{-3}$ | With cosine annealing |

## 6. Conclusion

Conditional GASP extends standard GASP to heterogeneous multi-tissue imaging by learning tissue-adaptive polynomial coefficients. The key insight is that multi-acquisition bSSFP signals implicitly encode tissue relaxation properties, which can be extracted by a neural encoder and used to predict optimal coefficients. This approach maintains compatibility with the GASP polynomial framework while dramatically improving performance across tissues with diverse $T_2/T_1$ ratios.

## References

1. Bangerter NK, et al. "Analysis of multiple-acquisition SSFP." Magn Reson Med. 2004.
2. Leupold J, et al. "Quantitative T2 mapping with spectrally selective SSFP." Magn Reson Med. 2008.
3. Zur Y, et al. "Spoiling of transverse magnetization in steady-state sequences." Magn Reson Med. 1991.

## Appendix: Tissue Parameters

| Tissue | $T_1$ (s) | $T_2$ (s) | $T_2/T_1$ |
|--------|-----------|-----------|-----------|
| Water/CSF | 4.0 | 2.0 | 0.50 |
| Gray Matter | 0.9 | 0.1 | 0.11 |
| White Matter | 0.6 | 0.08 | 0.13 |
| Muscle | 0.9 | 0.05 | 0.056 |
| Fat | 0.25 | 0.07 | 0.28 |
| Liver | 0.5 | 0.04 | 0.08 |
| Tendon | 0.4 | 0.005 | 0.0125 |

*Values approximate for 1.5T/3T field strengths.*
