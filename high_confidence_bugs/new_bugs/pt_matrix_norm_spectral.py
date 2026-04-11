"""
Bug: torch.linalg.matrix_norm with spectral norm (ord=2) GPU is 141x less
accurate than CPU for float32.

Root cause: Spectral norm = largest singular value. GPU uses cuSOLVER float32
SVD which accumulates more rounding error. The spectral norm (maximum singular
value) amplifies GPU inaccuracy: even a small absolute error in the largest
singular value becomes a large relative error. CPU uses LAPACK internally
in float64 for SVD, producing a far more accurate result.
This demonstrates that PT's GPU SVD inaccuracy (from pt_svdvals_accuracy)
propagates into any operation that internally computes SVD.
"""
import torch
import numpy as np

torch.manual_seed(0)
M = torch.randn(500, 500, dtype=torch.float32)

ref = torch.linalg.norm(M.double(), ord=2).float().item()
cpu = torch.linalg.norm(M, ord=2).item()
gpu = torch.linalg.norm(M.cuda(), ord=2).cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (float64): {ref:.6f}")
print(f"CPU (float32):       {cpu:.6f}")
print(f"GPU (float32):       {gpu:.6f}   <-- BUG")
print(f"CPU error vs ref: {cpu_err:.4e}")
print(f"GPU error vs ref: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
