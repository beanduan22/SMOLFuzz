"""
Bug #3 — INCONSISTENT: torch.sin CPU/GPU divergence at small scale (near zero)
Cluster: 11 models (23, 81, 137, 138, 151, 173, 174, 194, 205, 208, 220)
Representative: model_0173  ← smallest reproducer in the entire dataset (5 APIs)
Trigger: scale_small mutation (inputs multiplied by ~1e-6)

Root cause: For x near 0, sin(x) ≈ x, but CPU and GPU use different
polynomial approximations with different ULP errors. The absolute difference
is tiny but logsumexp (log of a sum of exponentials) amplifies relative
differences, making the discrepancy observable.
"""
import torch

torch.manual_seed(0)
x = torch.randn(4, 8) * 1e-4          # scale_small: push into near-zero regime

cpu_sin = torch.sin(x)
gpu_sin = torch.sin(x.cuda()).cpu()

cpu_out = torch.special.logsumexp(cpu_sin, dim=1)
gpu_out = torch.special.logsumexp(gpu_sin.cuda(), dim=1).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"sin diff before logsumexp: {(cpu_sin - gpu_sin).abs().max():.4e}")
print(f"max |cpu - gpu| after logsumexp: {diff.max():.4e}")
print(f"oracle tol=1e-06, divergence detected: {diff.max() > 1e-6}")
# Note: this bug is GPU architecture/driver dependent. It was observed during
# fuzzing with a specific CUDA device. On IEEE-754 strict GPUs the diff may be 0.
