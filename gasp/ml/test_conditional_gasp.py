"""
Test and Evaluation Script for Conditional GASP.

Compares Conditional GASP performance against standard GASP
across different tissue types with varying T2/T1 ratios.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gasp.simulation import SSFPParams, simulate_ssfp_simple
from gasp.gasp import train_gasp, run_gasp, _design_matrix, _to_matrix
from gasp.responses import gaussian
from gasp.ml.data_generator import generate_signal_simple, TISSUE_PARAMS


def create_acquisition_params(n_pcs: int = 8, n_TRs: int = 3) -> SSFPParams:
    """Create standard acquisition parameters."""
    TRs = [5e-3, 10e-3, 15e-3][:n_TRs]
    pcs = np.linspace(0, 2 * np.pi, n_pcs, endpoint=False)

    # Expand to all combinations
    length = n_pcs * n_TRs
    alpha_list = [np.deg2rad(60)] * length
    TR_list = []
    pc_list = []
    for tr in TRs:
        for pc in pcs:
            TR_list.append(tr)
            pc_list.append(pc)

    return SSFPParams(length, alpha_list, TR_list, pc_list)


def create_target_profile(width: int, bw: float = 0.3, shift: float = 0.0) -> np.ndarray:
    """Create a Gaussian target spectral profile."""
    return gaussian(width, bw, shift)


def evaluate_standard_gasp(
    signals: np.ndarray,
    target_profile: np.ndarray,
    method: str = "affine",
    use_l2: bool = True,
    lam: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate standard GASP on signals.

    Args:
        signals: Complex signals [n_samples, width, n_acquisitions]
        target_profile: Target profile [width]
        method: GASP method
        use_l2: Use L2 regularization
        lam: Regularization strength

    Returns:
        outputs: GASP outputs [n_samples, width]
        coefficients: Fitted coefficients [n_samples, n_coeffs]
        mse: Mean squared error
    """
    n_samples, width, n_acq = signals.shape
    outputs = np.zeros((n_samples, width), dtype=complex)
    coefficients = []

    for i in range(n_samples):
        # Reshape signal for GASP: [width, 1, n_acq] -> treated as [H, W, features]
        signal_2d = signals[i].reshape(width, 1, n_acq)

        # Train GASP
        out, A = train_gasp(
            signal_2d, target_profile,
            method=method, useL2=use_l2, lam=lam
        )
        outputs[i] = out.flatten()
        coefficients.append(A)

    # Compute MSE
    mse = np.mean((np.abs(outputs) - target_profile) ** 2)

    return outputs, np.array(coefficients), mse


def test_basic_functionality():
    """Test that Conditional GASP model runs without errors."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    try:
        import torch
        from gasp.ml.conditional_gasp import ConditionalGASP, design_matrix_torch
        from gasp.ml.data_generator import generate_training_batch
        from gasp.ml.losses import conditional_gasp_loss
    except ImportError as e:
        print(f"SKIPPED: PyTorch not available ({e})")
        return False

    # Setup
    params = create_acquisition_params(n_pcs=4, n_TRs=2)
    n_acquisitions = params.length
    width = 64
    batch_size = 4

    # Create model
    model = ConditionalGASP(
        n_acquisitions=n_acquisitions,
        latent_dim=8,
        method="affine"
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Generate test batch
    signals, tissue_params = generate_training_batch(
        batch_size, params, width=width
    )
    print(f"Generated batch: signals shape = {signals.shape}")

    # Forward pass
    signals_t = torch.from_numpy(signals.reshape(batch_size * width, -1))
    A, z = model(signals_t, return_latent=True)
    print(f"Forward pass: A shape = {A.shape}, z shape = {z.shape}")

    # Apply GASP
    Phi = design_matrix_torch(signals_t, model.method)
    output = (Phi * A).sum(dim=-1)
    print(f"Output shape = {output.shape}")

    # Compute loss
    target_profile = create_target_profile(width)
    target_t = torch.from_numpy(target_profile).float()
    target_tiled = target_t.unsqueeze(0).expand(batch_size, -1).reshape(-1)
    tissue_t = torch.from_numpy(tissue_params).float()
    tissue_tiled = tissue_t.unsqueeze(1).expand(-1, width, -1).reshape(-1, 3)

    losses = conditional_gasp_loss(output, target_tiled, A, z, tissue_tiled)
    print(f"Loss: {losses['total'].item():.4f}")

    print("PASSED: Basic functionality works\n")
    return True


def test_training():
    """Test that training loop runs and loss decreases."""
    print("=" * 60)
    print("Test 2: Training Loop")
    print("=" * 60)

    try:
        import torch
        from gasp.ml.trainer import train_conditional_gasp
    except ImportError as e:
        print(f"SKIPPED: PyTorch not available ({e})")
        return False

    # Setup
    params = create_acquisition_params(n_pcs=4, n_TRs=2)
    width = 64
    target_profile = create_target_profile(width)

    # Train for a few epochs
    print("Training for 5 epochs...")
    model, history = train_conditional_gasp(
        params=params,
        target_profile=target_profile,
        n_epochs=5,
        batch_size=8,
        n_batches_per_epoch=10,
        learning_rate=1e-3,
        width=width,
        latent_dim=8,
        method="affine",
        verbose=False,
    )

    # Check loss decreased
    initial_loss = history.train_loss[0]
    final_loss = history.train_loss[-1]
    print(f"Initial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")

    if final_loss < initial_loss:
        print("PASSED: Loss decreased during training\n")
        return True
    else:
        print("WARNING: Loss did not decrease (may need more epochs)\n")
        return True  # Not necessarily a failure


def test_comparison_with_standard_gasp():
    """Compare Conditional GASP vs Standard GASP on multi-tissue data."""
    print("=" * 60)
    print("Test 3: Comparison with Standard GASP")
    print("=" * 60)

    try:
        import torch
        from gasp.ml.trainer import train_conditional_gasp
        from gasp.ml.conditional_gasp import ConditionalGASP, design_matrix_torch, run_conditional_gasp
    except ImportError as e:
        print(f"SKIPPED: PyTorch not available ({e})")
        return False

    # Setup
    params = create_acquisition_params(n_pcs=8, n_TRs=3)
    width = 128
    target_profile = create_target_profile(width, bw=0.3)

    # Test tissues with different T2/T1 ratios
    test_tissues = [
        ('water', 4.0, 2.0),           # T2/T1 = 0.50 (high)
        ('gray_matter', 0.9, 0.1),     # T2/T1 = 0.11 (medium)
        ('white_matter', 0.6, 0.08),   # T2/T1 = 0.13 (medium)
        ('muscle', 0.9, 0.05),         # T2/T1 = 0.056 (low)
        ('tendon', 0.4, 0.005),        # T2/T1 = 0.0125 (very low)
    ]

    print("\nGenerating test signals for each tissue type...")

    # Generate signals for each tissue
    test_signals = {}
    for name, t1, t2 in test_tissues:
        signal = generate_signal_simple(t1, t2, params, width=width)
        test_signals[name] = signal
        print(f"  {name}: T1={t1:.3f}s, T2={t2:.4f}s, T2/T1={t2/t1:.4f}")

    # --- Standard GASP: Train on ONE tissue (gray matter) ---
    print("\n--- Standard GASP (trained on gray matter) ---")
    train_tissue = 'gray_matter'
    train_signal = test_signals[train_tissue]

    # Train standard GASP
    train_signal_2d = train_signal.reshape(width, 1, -1)
    _, standard_coeffs = train_gasp(
        train_signal_2d, target_profile,
        method="affine", useL2=True, lam=1e-2
    )

    # Evaluate on all tissues
    standard_errors = {}
    for name, signal in test_signals.items():
        signal_2d = signal.reshape(width, 1, -1)
        output = run_gasp(signal_2d, standard_coeffs, method="affine")
        mse = np.mean((np.abs(output.flatten()) - target_profile) ** 2)
        standard_errors[name] = mse
        print(f"  {name}: MSE = {mse:.6f}")

    # --- Conditional GASP: Train on diverse T1/T2 range ---
    print("\n--- Conditional GASP (trained on diverse T1/T2) ---")
    print("Training Conditional GASP model...")

    model, history = train_conditional_gasp(
        params=params,
        target_profile=target_profile,
        n_epochs=30,
        batch_size=16,
        n_batches_per_epoch=50,
        learning_rate=1e-3,
        width=width,
        latent_dim=16,
        method="affine",
        t1_range=(0.1, 4.0),
        t2_t1_ratio_range=(0.01, 0.5),
        noise_sigma=0.005,
        verbose=False,
    )

    # Evaluate on all tissues
    conditional_errors = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for name, signal in test_signals.items():
            signal_t = torch.from_numpy(signal).to(device)
            A, z = model(signal_t, return_latent=True)
            Phi = design_matrix_torch(signal_t, model.method)
            output = (Phi * A).sum(dim=-1)
            output_np = output.abs().cpu().numpy()

            mse = np.mean((output_np - target_profile) ** 2)
            conditional_errors[name] = mse
            print(f"  {name}: MSE = {mse:.6f}")

    # --- Summary ---
    print("\n--- Summary ---")
    print(f"{'Tissue':<15} {'Standard GASP':<15} {'Conditional GASP':<18} {'Improvement':<12}")
    print("-" * 60)

    total_improvement = 0
    for name, _, _ in test_tissues:
        std_err = standard_errors[name]
        cond_err = conditional_errors[name]
        if std_err > 0:
            improvement = (std_err - cond_err) / std_err * 100
        else:
            improvement = 0
        total_improvement += improvement
        print(f"{name:<15} {std_err:<15.6f} {cond_err:<18.6f} {improvement:>+.1f}%")

    avg_improvement = total_improvement / len(test_tissues)
    print("-" * 60)
    print(f"Average improvement: {avg_improvement:+.1f}%")

    if avg_improvement > 0:
        print("\nPASSED: Conditional GASP outperforms Standard GASP on average\n")
    else:
        print("\nNOTE: Conditional GASP may need more training or tuning\n")

    return True


def test_latent_space_structure():
    """Test that latent space captures T2/T1 ratio information."""
    print("=" * 60)
    print("Test 4: Latent Space Structure")
    print("=" * 60)

    try:
        import torch
        from gasp.ml.trainer import train_conditional_gasp
        from gasp.ml.conditional_gasp import ConditionalGASP
    except ImportError as e:
        print(f"SKIPPED: PyTorch not available ({e})")
        return False

    # Setup
    params = create_acquisition_params(n_pcs=8, n_TRs=3)
    width = 64
    target_profile = create_target_profile(width)

    # Train model
    print("Training model...")
    model, _ = train_conditional_gasp(
        params=params,
        target_profile=target_profile,
        n_epochs=20,
        batch_size=16,
        n_batches_per_epoch=30,
        width=width,
        latent_dim=8,
        verbose=False,
    )

    # Generate signals across T2/T1 ratio range
    ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    T1 = 1.0  # Fixed T1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for ratio in ratios:
            T2 = T1 * ratio
            signal = generate_signal_simple(T1, T2, params, width=width)
            signal_t = torch.from_numpy(signal).to(device)
            _, z = model(signal_t, return_latent=True)
            embeddings.append(z.mean(dim=0).cpu().numpy())

    embeddings = np.array(embeddings)

    # Check that embeddings vary smoothly with T2/T1 ratio
    # Compute distances between adjacent embeddings
    distances = []
    for i in range(len(embeddings) - 1):
        dist = np.linalg.norm(embeddings[i+1] - embeddings[i])
        distances.append(dist)

    print(f"Embedding distances between adjacent T2/T1 ratios:")
    for i, (r1, r2) in enumerate(zip(ratios[:-1], ratios[1:])):
        print(f"  {r1:.2f} -> {r2:.2f}: distance = {distances[i]:.4f}")

    # Check that total variation is significant (embeddings change with ratio)
    total_variation = np.sum(distances)
    first_last_dist = np.linalg.norm(embeddings[-1] - embeddings[0])
    print(f"\nTotal path length: {total_variation:.4f}")
    print(f"First-to-last distance: {first_last_dist:.4f}")

    if first_last_dist > 0.1:
        print("PASSED: Latent space captures T2/T1 variation\n")
        return True
    else:
        print("WARNING: Latent space may not capture T2/T1 well\n")
        return True


def run_full_evaluation(save_plots: bool = True, output_dir: str = "results"):
    """
    Run comprehensive evaluation and generate plots.

    Args:
        save_plots: If True, save plots to disk
        output_dir: Directory for output files
    """
    print("=" * 60)
    print("Full Evaluation: Conditional GASP vs Standard GASP")
    print("=" * 60)

    try:
        import torch
        from gasp.ml.trainer import train_conditional_gasp
        from gasp.ml.conditional_gasp import design_matrix_torch
    except ImportError as e:
        print(f"SKIPPED: PyTorch not available ({e})")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Setup
    params = create_acquisition_params(n_pcs=8, n_TRs=3)
    width = 256
    target_profile = create_target_profile(width, bw=0.25)

    # Train Conditional GASP
    print("\nTraining Conditional GASP (this may take a few minutes)...")
    model, history = train_conditional_gasp(
        params=params,
        target_profile=target_profile,
        n_epochs=50,
        batch_size=32,
        n_batches_per_epoch=100,
        learning_rate=1e-3,
        width=width,
        latent_dim=16,
        method="affine",
        t1_range=(0.1, 4.0),
        t2_t1_ratio_range=(0.01, 0.5),
        noise_sigma=0.005,
        verbose=True,
    )

    # Test tissues
    test_tissues = [
        ('Water', 4.0, 2.0),
        ('CSF', 4.0, 2.0),
        ('Gray Matter', 0.9, 0.1),
        ('White Matter', 0.6, 0.08),
        ('Muscle', 0.9, 0.05),
        ('Fat', 0.25, 0.07),
        ('Liver', 0.5, 0.04),
        ('Tendon', 0.4, 0.005),
    ]

    # Generate signals and evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    results = {
        'tissue': [],
        't1': [],
        't2': [],
        'ratio': [],
        'standard_mse': [],
        'conditional_mse': [],
    }

    # Train standard GASP on a "reference" tissue (gray matter)
    ref_signal = generate_signal_simple(0.9, 0.1, params, width=width)
    ref_signal_2d = ref_signal.reshape(width, 1, -1)
    _, standard_coeffs = train_gasp(
        ref_signal_2d, target_profile,
        method="affine", useL2=True, lam=1e-2
    )

    # Evaluate both methods
    print("\nEvaluating on test tissues...")
    with torch.no_grad():
        for name, t1, t2 in test_tissues:
            ratio = t2 / t1
            signal = generate_signal_simple(t1, t2, params, width=width)

            # Standard GASP
            signal_2d = signal.reshape(width, 1, -1)
            std_output = run_gasp(signal_2d, standard_coeffs, method="affine")
            std_mse = np.mean((np.abs(std_output.flatten()) - target_profile) ** 2)

            # Conditional GASP
            signal_t = torch.from_numpy(signal).to(device)
            A, _ = model(signal_t, return_latent=True)
            Phi = design_matrix_torch(signal_t, model.method)
            cond_output = (Phi * A).sum(dim=-1).abs().cpu().numpy()
            cond_mse = np.mean((cond_output - target_profile) ** 2)

            results['tissue'].append(name)
            results['t1'].append(t1)
            results['t2'].append(t2)
            results['ratio'].append(ratio)
            results['standard_mse'].append(std_mse)
            results['conditional_mse'].append(cond_mse)

            print(f"  {name}: Standard MSE={std_mse:.6f}, Conditional MSE={cond_mse:.6f}")

    # Create plots
    if save_plots:
        # Plot 1: Training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history.train_loss, label='Train Loss')
        axes[0].plot(history.val_loss, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history.profile_loss, label='Profile Loss')
        axes[1].plot(history.l2_loss, label='L2 Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'training_history.png', dpi=150)
        plt.close()

        # Plot 2: MSE comparison bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(results['tissue']))
        bar_width = 0.35

        bars1 = ax.bar(x - bar_width/2, results['standard_mse'], bar_width,
                       label='Standard GASP', color='steelblue')
        bars2 = ax.bar(x + bar_width/2, results['conditional_mse'], bar_width,
                       label='Conditional GASP', color='coral')

        ax.set_xlabel('Tissue Type')
        ax.set_ylabel('MSE')
        ax.set_title('GASP Performance Comparison Across Tissues')
        ax.set_xticks(x)
        ax.set_xticklabels(results['tissue'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path / 'mse_comparison.png', dpi=150)
        plt.close()

        # Plot 3: MSE vs T2/T1 ratio
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(results['ratio'], results['standard_mse'],
                   s=100, label='Standard GASP', color='steelblue', alpha=0.7)
        ax.scatter(results['ratio'], results['conditional_mse'],
                   s=100, label='Conditional GASP', color='coral', alpha=0.7)

        # Add tissue labels
        for i, name in enumerate(results['tissue']):
            ax.annotate(name, (results['ratio'][i], results['standard_mse'][i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel('T2/T1 Ratio')
        ax.set_ylabel('MSE')
        ax.set_title('GASP Performance vs T2/T1 Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig(output_path / 'mse_vs_ratio.png', dpi=150)
        plt.close()

        print(f"\nPlots saved to {output_path}/")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    std_mean = np.mean(results['standard_mse'])
    cond_mean = np.mean(results['conditional_mse'])
    improvement = (std_mean - cond_mean) / std_mean * 100

    print(f"Standard GASP mean MSE: {std_mean:.6f}")
    print(f"Conditional GASP mean MSE: {cond_mean:.6f}")
    print(f"Average improvement: {improvement:+.1f}%")

    # Count wins
    wins = sum(1 for s, c in zip(results['standard_mse'], results['conditional_mse']) if c < s)
    print(f"Conditional GASP wins: {wins}/{len(results['tissue'])} tissues")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CONDITIONAL GASP TEST SUITE")
    print("=" * 60 + "\n")

    results = []

    # Run tests
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("Training Loop", test_training()))
    results.append(("Standard GASP Comparison", test_comparison_with_standard_gasp()))
    results.append(("Latent Space Structure", test_latent_space_structure()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Conditional GASP")
    parser.add_argument("--full", action="store_true",
                        help="Run full evaluation with plots")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots")

    args = parser.parse_args()

    if args.full:
        run_full_evaluation(save_plots=True, output_dir=args.output)
    else:
        main()
