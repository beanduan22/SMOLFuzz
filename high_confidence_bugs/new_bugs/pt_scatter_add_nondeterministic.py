"""Bug 54: torch scatter_add_ float16 GPU is ~40x less accurate than CPU.
GPU uses non-deterministic atomicAdd; float16 non-associativity accumulates error.
"""
import torch

src = torch.ones(100000, dtype=torch.float16)
idx = torch.randint(50, (100000,))

ref = torch.zeros(50, dtype=torch.float64).scatter_add_(0, idx, src.double())
cpu = torch.zeros(50, dtype=torch.float16).scatter_add_(0, idx, src)
gpu = torch.zeros(50, dtype=torch.float16).cuda().scatter_add_(0, idx.cuda(), src.cuda()).cpu()

cpu_err = (cpu.double() - ref).abs().max().item()
gpu_err = (gpu.double() - ref).abs().max().item()

print(f"CPU error vs float64: {cpu_err:.3e}")
print(f"GPU error vs float64: {gpu_err:.3e}")   # BUG: ~40x larger
print(f"GPU/CPU ratio: {gpu_err / cpu_err:.1f}x")
assert gpu_err > 10 * cpu_err, "bug not reproduced"
print("BUG CONFIRMED: GPU float16 scatter_add error is significantly larger than CPU")
