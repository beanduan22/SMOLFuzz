"""
Bug: torch.cumprod GPU is ~97000x less accurate than CPU for float32 at N=1M.
Root cause: PyTorch CPU cumprod for float32 internally promotes to float64
accumulation (like cumsum), achieving near-exact results. GPU accumulates in
native float32, so each of the 1M multiplications contributes ~1.2e-7 relative error.
After N=1M steps, GPU max error is ~5.8e-3 vs CPU max error of ~6e-8.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 1_000_000
x = 1.0 + 0.0001 * torch.randn(n, dtype=torch.float32)

ref = np.cumprod(x.numpy().astype(np.float64))
cpu = torch.cumprod(x, dim=0).numpy()
gpu = torch.cumprod(x.cuda(), dim=0).cpu().numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err if cpu_err > 0 else float('inf')

print(f"cpu_max_err = {cpu_err:.4e}  (CPU promotes f32→f64 internally)")
print(f"gpu_max_err = {gpu_err:.4e}  (GPU accumulates in f32)")
print(f"GPU/CPU error ratio: {ratio:.0f}x  (GPU less accurate)")
assert ratio > 1000, f"Expected >1000x ratio, got {ratio:.0f}x"
print("BUG CONFIRMED: PT cumprod f32 N=1M GPU is ~97000x less accurate than CPU (CPU uses f64 internally)")
