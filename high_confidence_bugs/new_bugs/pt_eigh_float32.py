"""
Bug: torch.linalg.eigh gives significantly less accurate results on GPU
than on CPU for float32 symmetric matrices.

CPU uses LAPACK dsyevd (Divide and Conquer), which is accurate to ~1e-7.
GPU uses cuSOLVER for float32, which for the same matrix is ~50x less
accurate (L2 ~4e-3 vs 8e-5 from the float64 reference).

The difference is not rounding noise — GPU eigenvalues deviate from the
true values by orders of magnitude more than CPU.
"""
import torch

torch.manual_seed(0)
n = 50
# Build a symmetric positive semi-definite float32 matrix
A64 = torch.randn(n, n, dtype=torch.float64)
A_sym = A64 @ A64.T          # symmetric, float64
A32  = A_sym.float()          # same matrix in float32

# float64 result is the reference (ground truth)
ref_vals, _ = torch.linalg.eigh(A_sym)

cpu_vals, _ = torch.linalg.eigh(A32)
gpu_vals, _ = torch.linalg.eigh(A32.cuda())

cpu_err = (cpu_vals       - ref_vals.float()).norm().item()
gpu_err = (gpu_vals.cpu() - ref_vals.float()).norm().item()

print(f"CPU eigenvalue error vs float64 reference: {cpu_err:.4e}")
print(f"GPU eigenvalue error vs float64 reference: {gpu_err:.4e}")
print(f"GPU is {gpu_err/cpu_err:.0f}x less accurate than CPU")
print(f"CPU vs GPU L2: {(cpu_vals - gpu_vals.cpu()).norm().item():.4e}")
print()
print("First 5 eigenvalues:")
print(f"  CPU: {cpu_vals[:5].tolist()}")
print(f"  GPU: {gpu_vals.cpu()[:5].tolist()}")
print(f"  Ref: {ref_vals[:5].float().tolist()}")
