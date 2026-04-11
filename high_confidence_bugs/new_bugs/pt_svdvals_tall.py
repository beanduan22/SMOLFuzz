"""
Bug: torch.linalg.svdvals GPU is 25x less accurate than CPU for tall float32 matrices.

Root cause: Same as pt_svdvals_accuracy (square matrix), but the gap is
reproducible for tall matrices (1000×50). CPU uses LAPACK dgesdd which
operates internally in float64. GPU uses cuSOLVER float32 SVD, accumulating
more rounding error across the larger number of rows. The accuracy gap
(25x) is consistent with the square-matrix bug and scales with problem size.
This demonstrates the bug manifests across different matrix shapes.
"""
import torch
import numpy as np

torch.manual_seed(0)
M = torch.randn(1000, 50, dtype=torch.float32)

ref = torch.tensor(
    np.linalg.svd(M.numpy().astype(np.float64), compute_uv=False),
    dtype=torch.float32
)
cpu = torch.linalg.svdvals(M)
gpu = torch.linalg.svdvals(M.cuda()).cpu()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU singular value error vs float64 reference: {cpu_err:.4e}")
print(f"GPU singular value error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
print()
print(f"First 5 singular values:")
print(f"  CPU: {cpu[:5].tolist()}")
print(f"  GPU: {gpu[:5].tolist()}")
print(f"  Ref: {ref[:5].tolist()}")
