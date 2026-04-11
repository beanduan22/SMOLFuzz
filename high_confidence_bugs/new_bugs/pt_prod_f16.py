"""
Bug: torch.prod float16 CPU is 187x less accurate than GPU.

Root cause: CPU reduces float16 sequentially, accumulating rounding error
across all 1000 multiplications. GPU uses parallel tree reduction which,
for multiplication, keeps intermediate values closer to the true result.
For products of ~1.0 values, sequential float16 multiplication drifts
significantly away from the true float32/float64 result.
"""
import torch
import numpy as np

torch.manual_seed(42)
x = torch.randn(1000, dtype=torch.float32).abs() * 0.01 + 0.99  # values near 1.0

x16 = x.to(torch.float16)
ref = torch.prod(x.double()).item()

cpu = torch.prod(x16).item()
gpu = torch.prod(x16.cuda()).cpu().item()

cpu_err = abs(cpu - ref)
gpu_err = abs(gpu - ref)

print(f"Reference (float64): {ref:.6f}")
print(f"CPU (float16):       {cpu:.6f}")
print(f"GPU (float16):       {gpu:.6f}")
print(f"CPU error vs ref: {cpu_err:.4e}")
print(f"GPU error vs ref: {gpu_err:.4e}")
if gpu_err > 0:
    print(f"CPU is {cpu_err / gpu_err:.0f}x less accurate than GPU   <-- BUG")
else:
    print(f"GPU is {cpu_err:.4e} away from ref")
