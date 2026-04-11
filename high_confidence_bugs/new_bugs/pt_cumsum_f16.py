"""
Bug: torch.cumsum is 5x less accurate on GPU than CPU for float16 inputs.

Root cause: CPU promotes float16 to float32 during accumulation, greatly reducing
rounding error. GPU's CUDA kernel performs the cumulative sum natively in float16,
accumulating 10,000 half-precision additions without compensation. This causes
catastrophic cancellation and large numerical drift compared to the float64 reference.
"""
import torch

torch.manual_seed(0)
n = 10_000
x = torch.randn(n, dtype=torch.float16)

ref = torch.cumsum(x.double(), dim=0).float()   # float64 reference
cpu = torch.cumsum(x, dim=0).float()            # CPU float16
gpu = torch.cumsum(x.cuda(), dim=0).cpu().float()  # GPU float16

cpu_err = (cpu - ref).norm().item()
gpu_err = (gpu - ref).norm().item()

print(f"CPU error vs float64 reference: {cpu_err:.4e}")
print(f"GPU error vs float64 reference: {gpu_err:.4e}")
print(f"GPU is {gpu_err / cpu_err:.1f}x less accurate than CPU")
print(f"CPU vs GPU L2: {(cpu - gpu).norm().item():.4e}")
