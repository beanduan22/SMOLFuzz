"""
Bug #1 — INCONSISTENT: torch.sin CPU/GPU divergence at large scale
Cluster: 22 models (10, 22, 28, 45, 84, 86, 92, 104, 134, 152, 157, 160, 190, 196,
         199, 238, 258, 262, 267, 271, 276, 287, 296)
Representative: model_0196
Trigger: scale_large mutation (inputs multiplied by ~1000)

Root cause: GPU sin() uses a different polynomial range-reduction algorithm
than CPU. For large-magnitude inputs the range-reduction step introduces a
ULP-level rounding difference that propagates through subsequent operations.
"""
import torch

torch.manual_seed(0)
x = torch.randn(4, 8) * 1000.0          # scale_large: push into high-magnitude regime

cpu = torch.sin(x)
gpu = torch.sin(x.cuda()).cpu()

diff = (cpu - gpu).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
print(f"mean|cpu - gpu|: {diff.mean():.4e}")
print(f"oracle tol=1e-06, divergence detected: {diff.max() > 1e-6}")
# The fuzzer also caught this via the downstream model chain (erfc*sin→fc→i0e),
# which amplifies the per-element ULP difference to diffs > 0.2.
