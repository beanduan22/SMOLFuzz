"""
Bug: torch.linalg.svdvals GPU is 262x less accurate than CPU for complex64 matrices.

Root cause: CPU uses LAPACK `cgesdd` which internally promotes complex64
(float32 components) to higher precision. GPU uses cuSOLVER complex float32 SVD,
accumulating significantly more rounding error. For a 200×200 complex64 matrix,
GPU singular value errors are 262x larger than CPU errors vs the complex128 reference.

This is a complex-number version of pt_svdvals_accuracy (real float32) and
demonstrates the same underlying GPU SVD precision deficiency affects both
real and complex matrix types.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 200
M = torch.randn(n, n, dtype=torch.complex64)

ref = torch.linalg.svdvals(M.to(torch.complex128)).float()
cpu = torch.linalg.svdvals(M)
gpu = torch.linalg.svdvals(M.cuda()).cpu()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU singular value error vs complex128 reference: {cpu_err:.4e}")
print(f"GPU singular value error vs complex128 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
print()
print(f"First 5 singular values:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
