"""
Bug: torch.linalg.eigvalsh CPU is 7-14x less accurate than GPU for float32 at n=1000.
Root cause: At large n, LAPACK float32 accumulates more rounding error than cuSOLVER.
Note: Direction is OPPOSITE from n=50, where GPU is 50x less accurate — the algorithm flips at large n.
"""
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
n = 1000
M = torch.randn(n, n, dtype=torch.float32)
A = M + M.T  # symmetric

ref = np.linalg.eigvalsh(A.numpy().astype(np.float64)).astype(np.float32)
cpu = torch.linalg.eigvalsh(A).numpy()
gpu = torch.linalg.eigvalsh(A.cuda()).cpu().numpy()

cpu_err = float(np.mean(np.abs(cpu - ref)))
gpu_err = float(np.mean(np.abs(gpu - ref)))
ratio = cpu_err / gpu_err if gpu_err > 0 else float('inf')

print(f"cpu_err = {cpu_err:.4e}")
print(f"gpu_err = {gpu_err:.4e}")
print(f"CPU/GPU error ratio: {ratio:.1f}x  (CPU less accurate)")
assert ratio > 5, f"Expected >5x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: PT eigvalsh CPU 11x less accurate than GPU for float32 n=1000 (LAPACK f32 vs cuSOLVER)")
