"""
Bug: torch.linalg.matrix_norm nuclear norm GPU is 287x less accurate than CPU
for complex64 matrices.

Root cause: Nuclear norm = sum of singular values. GPU uses cuSOLVER complex
float32 SVD which accumulates more rounding error than CPU's LAPACK cgesdd.
The sum of 200 slightly-inaccurate singular values amplifies the per-value
error. This is the complex64 analogue of pt_matrix_norm_nuc (real float32)
and pt_svdvals_complex64, confirming the GPU SVD accuracy deficit extends
to complex number types.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 200
M = torch.randn(n, n, dtype=torch.complex64)

ref = np.linalg.norm(M.numpy().astype(np.complex128), "nuc")
cpu = torch.linalg.matrix_norm(M, ord="nuc").item()
gpu = torch.linalg.matrix_norm(M.cuda(), ord="nuc").cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (complex128): {ref:.6f}")
print(f"CPU (complex64):        {cpu:.6f}")
print(f"GPU (complex64):        {gpu:.6f}   <-- BUG")
print(f"CPU error vs reference: {cpu_err:.4e}")
print(f"GPU error vs reference: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
