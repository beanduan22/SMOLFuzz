"""Bug 56: torch.linalg.solve on near-singular system — massive CPU vs GPU divergence.
LAPACK (CPU) and cuSOLVER (GPU) use different pivot strategies.
"""
import torch
import numpy as np

# Vandermonde matrix: deterministically ill-conditioned for n=8
x = np.linspace(0, 1, 8, dtype=np.float32)
A = torch.tensor(np.vander(x, 8, increasing=True))
b = torch.ones(8, 1)

ref = torch.linalg.solve(A.double(), b.double()).float()
cpu = torch.linalg.solve(A, b)
gpu = torch.linalg.solve(A.cuda(), b.cuda()).cpu()

print(f"CPU error vs float64: {(cpu - ref).norm():.3e}")
print(f"GPU error vs float64: {(gpu - ref).norm():.3e}")
print(f"CPU vs GPU diff:      {(cpu - gpu).norm():.3e}")  # BUG: massive
assert (cpu - gpu).norm() > 1.0, "bug not reproduced"
print("BUG CONFIRMED: linalg.solve CPU vs GPU massive divergence on ill-conditioned matrix")
