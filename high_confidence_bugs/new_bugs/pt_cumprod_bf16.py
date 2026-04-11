"""
Bug: torch.cumprod bfloat16 GPU is 33x less accurate than CPU.

Root cause: CPU promotes bfloat16 to float32 for sequential multiplication.
GPU accumulates cumprod in native bfloat16 (7-bit mantissa), which
loses precision after many sequential multiplications. With 1000 elements,
the GPU result drifts 33x further from the float64 reference than CPU.
Bfloat16 has only 7 mantissa bits (vs float16's 10), making the gap even
worse per multiplication. The bug compounds because each step multiplies
the accumulated error by the next element.
"""
import torch
import numpy as np

torch.manual_seed(0)
x = torch.randn(1000, dtype=torch.bfloat16).abs() * 0.01 + 0.99

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
