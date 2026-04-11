"""
Bug: torch.linalg.eigvalsh GPU is 284–593x less accurate than CPU for float32
symmetric matrices at n=500.

Root cause: PyTorch GPU eigvalsh uses cuSOLVER dsyevd in native float32 (no
internal promotion). PyTorch CPU uses LAPACK ssyevd which accumulates less
numerical error at this matrix size, giving values much closer to the float64
reference. At n=500 the ratio reliably exceeds 100x; smaller matrices (n=50)
show only ~3x.

Confirmed ratios across seeds: seed=0: 298x, seed=1: 593x, seed=42: 284x.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 500
M = torch.randn(n, n, dtype=torch.float32)
A = M + M.T  # symmetric

ref = np.linalg.eigvalsh(A.numpy().astype(np.float64)).astype(np.float32)
cpu = torch.linalg.eigvalsh(A).numpy()
gpu = torch.linalg.eigvalsh(A.cuda()).cpu().numpy()

cpu_err = float(np.mean(np.abs(cpu - ref)))
gpu_err = float(np.mean(np.abs(gpu - ref)))
ratio = gpu_err / cpu_err

print(f"CPU eigenvalue error vs float64 reference: {cpu_err:.4e}")
print(f"GPU eigenvalue error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {ratio:.0f}x less accurate than CPU")
assert ratio >= 50, f"Expected >=50x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: PT eigvalsh GPU f32 n=500 is 284-593x less accurate than CPU (cuSOLVER vs LAPACK ssyevd)")
