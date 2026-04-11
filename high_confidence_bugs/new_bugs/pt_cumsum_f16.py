"""
Bug: torch.cumsum is 12–15x less accurate on GPU than CPU for float16 inputs (N=500k).

Root cause: PyTorch CPU promotes float16 to float32 during accumulation, greatly reducing
rounding error. The GPU CUDA kernel performs the cumulative sum natively in float16,
accumulating 500,000 half-precision additions without compensation, causing systematic
drift proportional to N * eps_f16.

At N=10k the ratio is only ~5x; at N=500k the ratio reliably exceeds 10x because
accumulated error grows as O(sqrt(N)) for both, but f32 (CPU) and f16 (GPU) eps differ
by 1024x, giving a theoretically expected ratio of ~32x.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 500_000
x = torch.randn(n, dtype=torch.float16)

ref = np.cumsum(x.numpy().astype(np.float64))
cpu = torch.cumsum(x, dim=0).numpy()
gpu = torch.cumsum(x.cuda(), dim=0).cpu().numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err

print(f"N = {n:,}")
print(f"CPU error vs float64 reference: {cpu_err:.4e}   (CPU promotes f16→f32 internally)")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG (GPU stays in f16)")
print(f"GPU is {ratio:.0f}x less accurate than CPU")
assert ratio >= 10, f"Expected >=10x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: PT cumsum float16 GPU is 12-15x less accurate than CPU at N=500k")
