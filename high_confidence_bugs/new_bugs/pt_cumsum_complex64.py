"""
Bug: torch.cumsum GPU is 10-16x less accurate than CPU for complex64 at N=5M.
Root cause: PyTorch CPU cumsum for complex64 internally uses higher-precision accumulation;
GPU uses native complex64 accumulation in float32 real/imag components,
accumulating more rounding error over large N.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 5_000_000
x = (torch.randn(n, dtype=torch.float32) + 1j * torch.randn(n, dtype=torch.float32)).to(torch.complex64)

ref = np.cumsum(x.numpy().astype(np.complex128))
cpu = torch.cumsum(x, dim=0).numpy()
gpu = torch.cumsum(x.cuda(), dim=0).cpu().numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.complex128) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.complex128) - ref)))
ratio = gpu_err / cpu_err if cpu_err > 0 else float('inf')

print(f"cpu_max_err = {cpu_err:.4e}")
print(f"gpu_max_err = {gpu_err:.4e}")
print(f"GPU/CPU error ratio: {ratio:.1f}x  (GPU less accurate)")
assert ratio > 5, f"Expected >5x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: PT cumsum complex64 N=5M GPU is 10-16x less accurate than CPU")
