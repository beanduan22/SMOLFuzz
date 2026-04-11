"""
Bug: torch.cumsum bfloat16 GPU is 4.6x less accurate than CPU.

Root cause: CPU promotes bfloat16 to float32 for cumulative sum, then
rounds back, giving results close to the float64 reference. GPU performs
cumulative sum natively in bfloat16 (7-bit mantissa), accumulating
10,000 additions without compensation and drifting significantly.
This is the opposite of the float16 cumsum bug (where GPU is more accurate).
"""
import torch
import numpy as np

torch.manual_seed(0)
x = torch.randn(10_000, dtype=torch.bfloat16)

ref = torch.cumsum(x.double(), dim=0).float()
cpu = torch.cumsum(x, dim=0).float()
gpu = torch.cumsum(x.cuda(), dim=0).cpu().float()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.1f}x less accurate than CPU")
print()
print(f"Last 5 cumulative sums:")
print(f"  ref: {ref[-5:].tolist()}")
print(f"  cpu: {cpu[-5:].tolist()}")
print(f"  gpu: {gpu[-5:].tolist()}")
