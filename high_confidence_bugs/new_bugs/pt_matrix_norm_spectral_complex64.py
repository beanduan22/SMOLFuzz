"""
Bug: torch.linalg.matrix_norm spectral norm (ord=2) GPU is 421x less accurate
than CPU for complex64 matrices.

Root cause: Spectral norm = largest singular value. GPU uses cuSOLVER complex
float32 SVD which accumulates more rounding error than CPU's LAPACK cgesdd.
The largest singular value amplifies GPU's SVD inaccuracy into a large relative
error for the spectral norm. CPU uses LAPACK internally in higher precision,
producing a result 421x closer to the complex128 reference.

This is the complex64 analogue of pt_matrix_norm_spectral (real float32, 141x gap).
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 200
M = torch.randn(n, n, dtype=torch.complex64)

ref = float(np.linalg.norm(M.numpy().astype(np.complex128), ord=2))
cpu = torch.linalg.matrix_norm(M, ord=2).item()
gpu = torch.linalg.matrix_norm(M.cuda(), ord=2).cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (complex128): {ref:.6f}")
print(f"CPU (complex64):        {cpu:.6f}")
print(f"GPU (complex64):        {gpu:.6f}   <-- BUG")
print(f"CPU error vs reference: {cpu_err:.4e}")
print(f"GPU error vs reference: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
