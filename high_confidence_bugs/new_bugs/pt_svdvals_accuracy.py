"""
Bug: torch.linalg.svdvals is 81x less accurate on GPU than CPU for float32.

Root cause: CPU uses LAPACK dgesdd (Divide & Conquer SVD, float64 internally).
GPU uses cuSOLVER's float32 SVD, which accumulates more rounding error.
The singular values from GPU deviate from the float64 reference by ~80x more
than CPU does.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 200
M = torch.randn(n, n, dtype=torch.float32)

ref = torch.tensor(np.linalg.svd(M.numpy().astype(np.float64), compute_uv=False),
                   dtype=torch.float32)

cpu = torch.linalg.svdvals(M)
gpu = torch.linalg.svdvals(M.cuda()).cpu()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU singular value error vs float64 reference: {cpu_err:.4e}")
print(f"GPU singular value error vs float64 reference: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
print(f"CPU vs GPU L2: {(cpu - gpu).norm().item():.4e}")
print()
print("First 5 singular values:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
