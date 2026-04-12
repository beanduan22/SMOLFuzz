"""
Bug: torch.linalg.slogdet (sign + log abs det) for near-singular float32 matrices.
CPU uses LAPACK's dgetrf LU decomposition; GPU uses cuSOLVER's dgetrf.
For matrices with very small (near-zero) determinants, the log abs det can differ
significantly between CPU and GPU due to different pivot strategies.
"""
import torch
import numpy as np

print("=== PyTorch linalg.slogdet near-singular: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

torch.manual_seed(42)
rng = np.random.default_rng(42)

def make_near_singular(n, min_eigenval):
    """Matrix with controlled smallest eigenvalue."""
    Q = torch.tensor(np.linalg.qr(rng.standard_normal((n, n)))[0], dtype=torch.float32)
    eigvals = torch.rand(n) + 1.0  # eigenvalues in [1, 2]
    eigvals[0] = min_eigenval
    return Q @ torch.diag(eigvals) @ Q.T

print("\n--- slogdet on near-singular matrices ---")
for n in [16, 32, 64]:
    for eps in [1e-3, 1e-5, 1e-7]:
        A = make_near_singular(n, eps)

        cpu_sign, cpu_logdet = torch.linalg.slogdet(A)
        gpu_sign, gpu_logdet = torch.linalg.slogdet(A.cuda())
        gpu_sign, gpu_logdet = gpu_sign.cpu(), gpu_logdet.cpu()

        # Reference: log|det| = sum of log|eigenvalues| = log(eps) + (n-1)*log(~1.5)
        # The logdet should be near log(eps) which is large negative
        logdet_diff = abs(cpu_logdet.item() - gpu_logdet.item())

        print(f"n={n}, eps={eps:.0e}: CPU_logdet={cpu_logdet:.4f}, GPU_logdet={gpu_logdet:.4f}, diff={logdet_diff:.4e}", end="")
        if logdet_diff > 0.1:
            print(f"  *** DIVERGENCE ***", end="")
        print()

# Very small determinant (near zero)
print("\n--- slogdet with very small determinant ---")
for n in [32, 64, 128]:
    for target_logdet in [-50, -100, -200]:
        # Create matrix where logdet should be ~target_logdet
        min_eig = np.exp(target_logdet - (n-1) * np.log(1.5))
        if min_eig < 1e-38:  # float32 min
            min_eig = 1e-38

        A = make_near_singular(n, float(min_eig))
        try:
            cpu_sign, cpu_logdet = torch.linalg.slogdet(A)
            gpu_sign, gpu_logdet = torch.linalg.slogdet(A.cuda())
            gpu_sign, gpu_logdet = gpu_sign.cpu(), gpu_logdet.cpu()

            diff = abs(cpu_logdet.item() - gpu_logdet.item())
            print(f"n={n}, target_logdet={target_logdet}: CPU={cpu_logdet:.2f}, GPU={gpu_logdet:.2f}, diff={diff:.4e}", end="")
            if diff > 1.0:
                print(f"  *** LOGDET DIVERGENCE ***", end="")
            print()
        except Exception as e:
            print(f"n={n}, target_logdet={target_logdet}: ERROR: {e}")

# Batch slogdet
print("\n--- Batch slogdet near-singular ---")
batch_size = 32
n = 32
for eps in [1e-5, 1e-7]:
    As = torch.stack([make_near_singular(n, eps) for _ in range(batch_size)])
    try:
        cpu_signs, cpu_logdets = torch.linalg.slogdet(As)
        gpu_signs, gpu_logdets = torch.linalg.slogdet(As.cuda())
        gpu_signs, gpu_logdets = gpu_signs.cpu(), gpu_logdets.cpu()

        diff = (cpu_logdets - gpu_logdets).abs().max().item()
        print(f"batch={batch_size}, n={n}, eps={eps:.0e}: max logdet diff={diff:.4e}", end="")
        if diff > 0.5:
            print(f"  *** BATCH DIVERGENCE ***", end="")
        print()
    except Exception as e:
        print(f"batch eps={eps:.0e}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
# Use batch slogdet with near-zero eigenvalue that reliably shows divergence
rng2 = np.random.default_rng(7)
n = 32
eps = 1e-7

def make_near_singular2(n, min_eigenval, seed_rng):
    Q = torch.tensor(np.linalg.qr(seed_rng.standard_normal((n, n)))[0], dtype=torch.float32)
    eigvals = torch.rand(n) + 1.0
    eigvals[0] = min_eigenval
    return Q @ torch.diag(eigvals) @ Q.T

batch_size = 16
As = torch.stack([make_near_singular2(n, eps, rng2) for _ in range(batch_size)])
try:
    cpu_signs, cpu_logdets = torch.linalg.slogdet(As)
    gpu_signs, gpu_logdets = torch.linalg.slogdet(As.cuda())
    gpu_signs, gpu_logdets = gpu_signs.cpu(), gpu_logdets.cpu()

    diff = (cpu_logdets - gpu_logdets).abs().max().item()
    print(f"slogdet(batch={batch_size}, {n}x{n} float32, min_eigenval=1e-7):")
    print(f"CPU logdets[:4]: {cpu_logdets[:4].tolist()}")
    print(f"GPU logdets[:4]: {gpu_logdets[:4].tolist()}")
    print(f"Max logdet diff: {diff:.4e}")
    if diff > 0.1:
        print(f"*** BUG: CPU and GPU log-determinants differ by up to {diff:.4e} ***")
    else:
        print(f"No significant divergence (diff={diff:.4e})")
except Exception as e:
    print(f"ERROR: {e}")
