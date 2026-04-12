"""
Bug: torch.linalg.inv on near-singular float32 matrices.
CPU uses LAPACK's dgetrf with partial pivoting; GPU uses cuSOLVER's dgetrf.
For matrices with very small eigenvalues, different pivot strategies cause
the computed inverses to diverge dramatically.
"""
import torch
import numpy as np

print("=== PyTorch linalg.inv near-singular: CPU vs GPU ===")
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("No CUDA"); exit(0)

rng = np.random.default_rng(42)

def make_near_singular(n, min_eigenval, seed_rng):
    Q = torch.tensor(np.linalg.qr(seed_rng.standard_normal((n, n)))[0], dtype=torch.float32)
    eigvals = torch.rand(n) + 1.0
    eigvals[0] = float(min_eigenval)
    return Q @ torch.diag(eigvals) @ Q.T

print("\n--- linalg.inv near-singular matrices ---")
for n in [32, 64, 128]:
    for eps in [1e-6, 1e-7, 1e-8]:
        A = make_near_singular(n, eps, rng)
        try:
            cpu_inv = torch.linalg.inv(A)
            gpu_inv = torch.linalg.inv(A.cuda()).cpu()

            diff = float((cpu_inv - gpu_inv).abs().max().item())
            rel_diff = diff / (cpu_inv.abs().max().item() + 1e-10)
            print(f"n={n}, eps={eps:.0e}: diff={diff:.3e}, rel_diff={rel_diff:.3e}", end="")
            if rel_diff > 0.01:
                print(f"  *** DIVERGENCE ***", end="")
            print()
        except Exception as e:
            print(f"n={n}, eps={eps:.0e}: ERROR: {e}")

# Verify inv(A) @ A ≈ I residual
print("\n--- Identity residual: ||inv(A) @ A - I|| ---")
for n in [64, 128]:
    for eps in [1e-7, 1e-8]:
        A = make_near_singular(n, eps, rng)
        try:
            cpu_inv = torch.linalg.inv(A)
            gpu_inv = torch.linalg.inv(A.cuda()).cpu()

            eye = torch.eye(n)
            cpu_res = float((cpu_inv @ A - eye).abs().max().item())
            gpu_res = float((gpu_inv @ A - eye).abs().max().item())
            print(f"n={n}, eps={eps:.0e}: CPU_residual={cpu_res:.3e}, GPU_residual={gpu_res:.3e}", end="")
            if abs(cpu_res - gpu_res) > 0.01:
                print(f"  *** RESIDUAL DIVERGE ***", end="")
            print()
        except Exception as e:
            print(f"n={n}, eps={eps:.0e}: ERROR: {e}")

print(f"\n=== BUG SUMMARY ===")
rng2 = np.random.default_rng(7)
n, eps = 128, 1e-7
A = make_near_singular(n, eps, rng2)
try:
    cpu_inv = torch.linalg.inv(A)
    gpu_inv = torch.linalg.inv(A.cuda()).cpu()

    diff = float((cpu_inv - gpu_inv).abs().max().item())
    rel_diff = diff / (cpu_inv.abs().max().item() + 1e-10)
    print(f"linalg.inv({n}x{n} float32, min_eigenval=1e-7):")
    print(f"CPU inv max val: {cpu_inv.abs().max().item():.3e}")
    print(f"GPU inv max val: {gpu_inv.abs().max().item():.3e}")
    print(f"Max element diff: {diff:.4e}, rel_diff: {rel_diff:.4e}")
    if rel_diff > 0.01:
        print(f"*** BUG: CPU and GPU matrix inverses differ by {rel_diff:.2e} relative ***")
    else:
        print(f"No significant divergence (rel_diff={rel_diff:.4e})")
except Exception as e:
    print(f"ERROR: {e}")
