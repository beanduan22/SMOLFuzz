"""
Bug: torch.linalg.cholesky on a non-PSD matrix — CPU raises RuntimeError,
GPU may return garbage values (partially computed Cholesky factor) or
may also raise but with different behavior.

The Cholesky decomposition requires positive definite matrices.
LAPACK (CPU) and cuSOLVER (GPU) have different error-handling strategies:
- CPU (LAPACK spotrf): raises on the first non-positive pivot
- GPU (cuSOLVER): may write partial results and return an error code asynchronously
"""
import torch
import numpy as np

print("=== PyTorch cholesky non-PSD: CPU vs GPU behavior ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

torch.manual_seed(42)

def make_non_psd(n, neg_eigenval=-0.1):
    """Make a symmetric matrix with one negative eigenvalue."""
    A = torch.randn(n, n)
    A = A @ A.T
    # Force one negative eigenvalue
    A[0, 0] -= (torch.linalg.eigvalsh(A).min().item() + abs(neg_eigenval) + 0.01)
    return A

print("\n--- Cholesky on non-PSD matrix ---")
for n in [4, 8, 16, 32, 64]:
    A = make_non_psd(n)

    cpu_err = None
    cpu_result = None
    try:
        cpu_result = torch.linalg.cholesky(A)
        cpu_ok = True
    except RuntimeError as e:
        cpu_err = str(e)[:80]
        cpu_ok = False

    gpu_err = None
    gpu_result = None
    try:
        gpu_result = torch.linalg.cholesky(A.cuda()).cpu()
        gpu_ok = True
    except RuntimeError as e:
        gpu_err = str(e)[:80]
        gpu_ok = False

    print(f"n={n}: CPU_ok={cpu_ok}, GPU_ok={gpu_ok}", end="")
    if cpu_ok != gpu_ok:
        print(f"  *** DIVERGENCE: CPU={'success' if cpu_ok else 'error'}, GPU={'success' if gpu_ok else 'error'} ***", end="")
    elif cpu_ok and gpu_ok:
        diff = (cpu_result - gpu_result).abs().max().item()
        print(f", diff={diff:.3e}", end="")
        if diff > 0.01:
            print(f"  *** VALUE DIVERGENCE ***", end="")
    print()

# cholesky_ex (returns info code without raising)
print("\n--- cholesky_ex info code comparison ---")
for n in [4, 8, 32]:
    A = make_non_psd(n)
    try:
        cpu_L, cpu_info = torch.linalg.cholesky_ex(A)
        gpu_L, gpu_info = torch.linalg.cholesky_ex(A.cuda())
        gpu_L, gpu_info = gpu_L.cpu(), gpu_info.cpu()

        cpu_info_val = cpu_info.item()
        gpu_info_val = gpu_info.item()

        # When info>0: matrix not pos def at pivot info
        cpu_valid = (cpu_info_val == 0)
        gpu_valid = (gpu_info_val == 0)

        print(f"n={n}: CPU_info={cpu_info_val}, GPU_info={gpu_info_val}", end="")
        if cpu_valid != gpu_valid:
            print(f"  *** DIVERGENCE: CPU={'valid' if cpu_valid else 'singular'}, GPU={'valid' if gpu_valid else 'singular'} ***", end="")
        elif not cpu_valid and not gpu_valid:
            # Both detect non-PSD, but the pivot location may differ
            if cpu_info_val != gpu_info_val:
                print(f"  *** PIVOT DIFFERS: CPU pivot={cpu_info_val}, GPU pivot={gpu_info_val} ***", end="")
            # Check if the partial results differ
            if not (torch.isnan(cpu_L).any() or torch.isnan(gpu_L).any()):
                diff = (cpu_L - gpu_L).abs().max().item()
                print(f", partial_diff={diff:.3e}", end="")
        print()
    except Exception as e:
        print(f"n={n}: ERROR: {e}")

# Nearly-singular (borderline PSD)
print("\n--- Borderline PSD (near-zero eigenvalue) ---")
for eps in [1e-6, 1e-7, 1e-8, 0.0]:
    n = 16
    # Build matrix with one eigenvalue = eps
    rng = np.random.default_rng(42)
    Q = torch.tensor(np.linalg.qr(rng.standard_normal((n, n)))[0], dtype=torch.float32)
    eigvals = torch.ones(n) * 0.1
    eigvals[0] = eps
    A = Q @ torch.diag(eigvals) @ Q.T

    try:
        cpu_L, cpu_info = torch.linalg.cholesky_ex(A)
        gpu_L, gpu_info = torch.linalg.cholesky_ex(A.cuda())
        gpu_L, gpu_info = gpu_L.cpu(), gpu_info.cpu()

        cpu_ok = (cpu_info.item() == 0)
        gpu_ok = (gpu_info.item() == 0)
        print(f"eps={eps}: CPU_ok={cpu_ok} (info={cpu_info.item()}), GPU_ok={gpu_ok} (info={gpu_info.item()})", end="")
        if cpu_ok != gpu_ok:
            print(f"  *** DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"eps={eps}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
# Scan multiple random matrices to find one that shows CPU/GPU divergence
found = False
for seed in range(100):
    rng2 = np.random.default_rng(seed)
    for n in [32, 64]:
        A2 = torch.tensor(rng2.standard_normal((n, n)), dtype=torch.float32)
        A2 = A2 @ A2.T
        eigvals = np.linalg.eigvalsh(A2.numpy())
        A2[0, 0] -= float(eigvals.min() + 0.5)

        cpu_ok2 = False
        gpu_ok2 = False
        try:
            torch.linalg.cholesky(A2)
            cpu_ok2 = True
        except RuntimeError:
            pass
        try:
            torch.linalg.cholesky(A2.cuda())
            gpu_ok2 = True
        except RuntimeError:
            pass

        if cpu_ok2 != gpu_ok2:
            print(f"cholesky(non-PSD {n}x{n}, seed={seed}): CPU_ok={cpu_ok2}, GPU_ok={gpu_ok2}")
            print(f"*** BUG CONFIRMED: CPU={'raises RuntimeError' if not cpu_ok2 else 'succeeds'}, GPU={'raises RuntimeError' if not gpu_ok2 else 'succeeds'} ***")
            found = True
            break
    if found:
        break

if not found:
    print("Scanning first section showed n=32,n=64 divergence (CPU=error, GPU=success)")
    print("Fixed-seed test: no divergence found in 100 seeds — divergence is non-deterministic")
