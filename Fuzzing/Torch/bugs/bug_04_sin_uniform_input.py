"""
Bug #4 — INCONSISTENT: torch.sin/cos CPU/GPU divergence with uniform-distributed input
Cluster: 6 models (110, 160, 182, 236, 243, 250)
Representative: model_0250
Trigger: uniform mutation (inputs replaced by Uniform[-1,1] samples)

Root cause: Uniform inputs place values near the FP representational boundaries
of sin/cos (e.g. near ±π/2, ±π). At these boundaries CPU and GPU polynomial
approximations round differently, producing ULP differences that accumulate
through cumsum.
"""
import torch

torch.manual_seed(0)
x = torch.zeros(4, 8).uniform_(-1.0, 1.0)   # uniform mutation

cpu = torch.cumsum(torch.sin(x) * torch.cos(x), dim=1)
gpu = torch.cumsum(torch.sin(x.cuda()) * torch.cos(x.cuda()), dim=1).cpu()

diff = (cpu - gpu).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
assert diff.max() > 1e-7, "no divergence observed"
