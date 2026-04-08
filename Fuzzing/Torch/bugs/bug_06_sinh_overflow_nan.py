"""
Bug #6 — NAN: torch.sinh overflow to inf/NaN at large scale
Cluster: 3 models (34, 93, 222)
Representative: model_0222  ← smallest NAN reproducer
Trigger: scale_large mutation

Root cause: sinh(x) = (e^x - e^{-x})/2 grows exponentially. For float32 inputs
with magnitude > ~89, sinh overflows to ±inf. Subsequent operations (trunc,
matmul, norm) propagate inf → NaN.  The CPU and GPU may differ in *which*
elements produce NaN vs inf, making this both a NAN bug and CPU/GPU inconsistency.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

x = torch.randn(4, 8) * 100.0    # scale_large: sinh overflows float32 here

with torch.no_grad():
    y = torch.sinh(x)             # overflow: values > 89 → ±inf
    z = fc1(y)
    w = torch.trunc(z)
    t = fc2(w)
    u = torch.sin(t) * torch.cos(t)
    v = torch.norm(u, dim=1, keepdim=True)

print(f"sinh range: [{y.min():.2e}, {y.max():.2e}]")
print(f"contains inf: {torch.isinf(y).any().item()}")
print(f"output contains NaN: {torch.isnan(v).any().item()}")
print(f"output contains inf: {torch.isinf(v).any().item()}")

# Minimal 3-line version:
x_large = torch.tensor([100.0, 200.0, -100.0])
print(f"\ntorch.sinh([100, 200, -100]) = {torch.sinh(x_large)}")
# expected: [inf, inf, -inf]
