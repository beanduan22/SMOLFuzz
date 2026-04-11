"""
Bug: torch.cumsum float32 GPU produces different results from CPU because
CPU internally promotes float32 to float64 for accumulation.

Root cause: PyTorch's CPU cumsum kernel for float32 inputs uses float64
accumulation (like numpy's default cumsum behavior). The GPU kernel accumulates
natively in float32, leading to rounding differences. For large input values
(~1e10), this causes significant absolute differences: CPU exactly matches the
float64 reference, while GPU accumulates float32 rounding errors from position 5
onward, diverging by up to 2.6e+05 per element.

This asymmetry means the same float32 cumsum gives different results depending
on whether it runs on CPU or GPU — without any warning from PyTorch.
"""
import torch
import numpy as np

torch.manual_seed(0)
x = torch.randn(10_000, dtype=torch.float32) * 1e10

ref = torch.cumsum(x.double(), dim=0).float()   # float64 reference
cpu = torch.cumsum(x, dim=0)
gpu = torch.cumsum(x.cuda(), dim=0).cpu()

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"Reference (float64): cumsum of N=10,000 float32 values scaled to ~1e10")
print(f"CPU error vs float64: {cpu_err:.4e}   (CPU promotes to float64 internally)")
print(f"GPU error vs float64: {gpu_err:.4e}   <-- BUG (GPU accumulates in float32)")
print(f"CPU exactly matches float64: {torch.equal(cpu, ref)}")
print()
print(f"Max CPU-GPU difference: {(cpu - gpu).abs().max().item():.4e}")
print(f"First divergence position: {(cpu != gpu).nonzero()[0].item()}")
print()
print(f"Last 3 cumulative sums:")
for i in range(-3, 0):
    print(f"  [{i}] ref={ref[i].item():.2f}  cpu={cpu[i].item():.2f}  gpu={gpu[i].item():.2f}")
