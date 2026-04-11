"""
Bug: torch.linalg.lstsq GPU returns completely wrong solution for rank-deficient
complex64 systems.

Root cause: Same as pt_lstsq_rankdef (real float32): CPU uses LAPACK's cgelsd
(complex SVD-based, handles rank deficiency). GPU uses cuBLAS cgels (QR-based,
assumes full rank). For the rank-deficient 3×3 complex Vandermonde-like system,
GPU returns a solution with L2 error >5e+6 vs the pseudoinverse reference,
while CPU matches the reference to ~5e-7.

This confirms the lstsq rank-deficiency bug affects complex number types too.
"""
import torch
import numpy as np

torch.manual_seed(0)
A = torch.tensor([
    [1.+0j, 2.+0j, 3.+0j],
    [4.+0j, 5.+0j, 6.+0j],
    [7.+0j, 8.+0j, 9.+0j],
], dtype=torch.complex64)

b = (torch.randn(3, 1) + 1j * torch.randn(3, 1)).to(torch.complex64)

ref = np.linalg.lstsq(
    A.numpy().astype(np.complex128),
    b.numpy().astype(np.complex128),
    rcond=None
)[0]

cpu = torch.linalg.lstsq(A, b).solution.numpy()
gpu = torch.linalg.lstsq(A.cuda(), b.cuda()).solution.cpu().numpy()

cpu_err = np.linalg.norm(cpu - ref)
gpu_err = np.linalg.norm(gpu - ref)

print(f"Reference (numpy complex128): {ref.flatten().tolist()}")
print(f"CPU (complex64): {cpu.flatten().tolist()}")
print(f"GPU (complex64): {gpu.flatten().tolist()}")
print()
print(f"CPU error vs reference: {cpu_err:.4e}")
print(f"GPU error vs reference: {gpu_err:.4e}   <-- BUG")
