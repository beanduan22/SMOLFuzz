"""
Bug: torch.prod bfloat16 CPU is 10x less accurate than GPU.

Root cause: CPU performs sequential multiplication in bfloat16 (7-bit mantissa),
accumulating rounding error across all 1000 multiplications. GPU uses parallel
tree reduction: it multiplies pairs, then pairs-of-pairs, reducing the depth
from N to log2(N) steps. With only 7 mantissa bits in bfloat16, the sequential
CPU approach loses more precision. The GPU's tree reduction has fewer precision-
losing steps and produces a result 10x closer to the float64 reference.
This is consistent with pt_prod_f16 (float16, CPU 131x less accurate).
"""
import torch
import numpy as np

torch.manual_seed(42)
x = torch.randn(1000, dtype=torch.float32).abs() * 0.01 + 0.99  # values near 1.0
x_bf16 = x.to(torch.bfloat16)

ref = torch.prod(x.double()).item()
cpu = torch.prod(x_bf16).item()
gpu = torch.prod(x_bf16.cuda()).cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (float64): {ref:.6f}")
print(f"CPU (bfloat16):      {cpu:.6f}   <-- BUG")
print(f"GPU (bfloat16):      {gpu:.6f}")
print(f"CPU error vs ref: {cpu_err:.4e}")
print(f"GPU error vs ref: {gpu_err:.4e}")
print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU")
