"""
Bug: torch.std returns inf on GPU but correct value on CPU for large float32 inputs.

Root cause: GPU uses a single-pass variance algorithm that computes sum(x^2) - n*mean^2.
For values ~1e20, x^2 overflows float32 to inf. CPU uses a two-pass or compensated
algorithm that avoids the overflow.

Reference (float64): ~1e19
CPU (float32):       ~1e19  (correct)
GPU (float32):       inf    <-- BUG
"""
import torch

torch.manual_seed(0)
x = torch.randn(1000, dtype=torch.float32) * 1e19 + 1e20

ref = torch.std(x.double()).item()
cpu = torch.std(x).item()
gpu = torch.std(x.cuda()).cpu().item()

print(f"Reference (float64): {ref:.4e}")
print(f"CPU (float32):       {cpu:.4e}")
print(f"GPU (float32):       {gpu}   <-- {'BUG' if not __import__('math').isinf(cpu) and __import__('math').isinf(gpu) else 'ok'}")
