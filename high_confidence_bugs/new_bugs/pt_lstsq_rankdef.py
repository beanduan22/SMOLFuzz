"""
Bug: torch.linalg.lstsq gives wrong result on GPU for rank-deficient matrices.

CPU uses LAPACK (gelsd/gelsy) which correctly handles rank deficiency via SVD.
GPU uses cuBLAS gels which only handles full-rank systems; silently returns
garbage for rank-deficient input instead of raising an error.

Related: https://github.com/pytorch/pytorch/issues/88101
"""
import torch
import numpy as np

# Rank-deficient matrix: rows are linearly dependent (rank=2, not 3)
A = torch.tensor([
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.],
], dtype=torch.float32)
B = A.clone()

# Reference: NumPy uses LAPACK gelsd (handles rank deficiency correctly)
ref = np.linalg.lstsq(A.numpy(), B.numpy(), rcond=None)[0]

cpu = torch.linalg.lstsq(A, B).solution
gpu = torch.linalg.lstsq(A.cuda(), B.cuda()).solution.cpu()

print("NumPy (reference):", ref[0])
print("CPU:              ", cpu[0].numpy())
print("GPU:              ", gpu[0].numpy())
print(f"CPU vs NumPy L2: {np.linalg.norm(ref - cpu.numpy()):.4e}")
print(f"GPU vs NumPy L2: {np.linalg.norm(ref - gpu.numpy()):.4e}")
