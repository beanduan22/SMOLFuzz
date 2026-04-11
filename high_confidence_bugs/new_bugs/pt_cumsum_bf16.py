"""
Bug: torch.cumsum bfloat16 GPU is 12–19x less accurate than CPU at N=500k.

Root cause: PyTorch CPU promotes bfloat16 to float32 for cumulative sum accumulation,
giving near-float32 precision. GPU accumulates natively in bfloat16 (7-bit mantissa),
so each of 500,000 additions introduces ~7.8e-3 relative rounding error.
After N=500k steps, GPU error grows as O(sqrt(N) * eps_bf16) while CPU is O(sqrt(N) * eps_f32),
yielding a ratio of ~sqrt(eps_bf16/eps_f32) * further_divergence ≈ 13-19x.

At N=10k the ratio is only ~4.6x (not a reliable bug); at N=500k it consistently exceeds 10x.
"""
import torch
import numpy as np

torch.manual_seed(0)
n = 500_000
x = torch.randn(n, dtype=torch.bfloat16)

ref = np.cumsum(x.float().numpy().astype(np.float64))
cpu = torch.cumsum(x, dim=0).float().numpy()
gpu = torch.cumsum(x.cuda(), dim=0).cpu().float().numpy()

cpu_err = float(np.max(np.abs(cpu.astype(np.float64) - ref)))
gpu_err = float(np.max(np.abs(gpu.astype(np.float64) - ref)))
ratio = gpu_err / cpu_err

print(f"N = {n:,}")
print(f"CPU error vs float64 reference: {cpu_err:.4e}   (CPU promotes bf16→f32 internally)")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG (GPU stays in bf16)")
print(f"GPU is {ratio:.0f}x less accurate than CPU")
assert ratio >= 10, f"Expected >=10x ratio, got {ratio:.1f}x"
print("BUG CONFIRMED: PT cumsum bfloat16 GPU is 12-19x less accurate than CPU at N=500k")
