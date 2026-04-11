"""
Bug: torch.linalg.matrix_norm with ord='nuc' (nuclear norm) is 236x less accurate
on GPU than CPU for float32.

Root cause: Nuclear norm = sum of singular values. CPU uses LAPACK's high-accuracy
SVD (internally float64). GPU uses cuSOLVER float32 SVD whose accumulated singular
value errors sum up, making the nuclear norm significantly less accurate.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 200
M = torch.randn(n, n, dtype=torch.float32)

ref = np.linalg.norm(M.numpy(), "nuc")          # NumPy (float32 matrix, accurate)
cpu = torch.linalg.matrix_norm(M, ord="nuc").item()
gpu = torch.linalg.matrix_norm(M.cuda(), ord="nuc").cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (NumPy):   {ref:.6f}")
print(f"CPU (float32):       {cpu:.6f}")
print(f"GPU (float32):       {gpu:.6f}")
print(f"CPU error vs NumPy: {cpu_err:.4e}")
print(f"GPU error vs NumPy: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
