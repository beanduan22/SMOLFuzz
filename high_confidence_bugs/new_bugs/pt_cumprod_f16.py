"""
Bug: torch.cumprod float16 GPU is 52x less accurate than CPU.

Root cause: CPU promotes float16 to float32 for cumulative product, then
rounds back to float16 after each step. GPU performs cumprod natively in
float16, accumulating 1000 multiplications without precision promotion.
The float16 mantissa (10 bits) loses precision across 1000 sequential
float16 multiplications, causing significant drift from the float64 reference.
Note: this is opposite of torch.prod f16, where GPU tree reduction is more accurate.
"""
import torch
import numpy as np

torch.manual_seed(0)
x = torch.randn(1000, dtype=torch.float16).abs() * 0.01 + 0.99

ref = torch.cumprod(x.double(), dim=0).float()
cpu = torch.cumprod(x, dim=0).float()
gpu = torch.cumprod(x.cuda(), dim=0).float().cpu()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}   <-- BUG")
print(f"GPU is {gpu_err / cpu_err:.0f}x less accurate than CPU")
print()
print(f"Last 5 cumulative products:")
print(f"  ref: {ref[-5:].tolist()}")
print(f"  cpu: {cpu[-5:].tolist()}")
print(f"  gpu: {gpu[-5:].tolist()}")
