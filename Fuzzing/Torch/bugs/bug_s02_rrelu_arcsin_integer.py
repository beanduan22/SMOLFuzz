"""
Bug S02 — INCONSISTENT: arcsin_ applied to integer-modded values (model_0039)
Trigger: baseline

Root cause: argsort returns integer indices [0,7]. remainder_(3) maps them to
{0,1,2}. Cast to float gives {0.0, 1.0, 2.0}. arcsin_(2.0) is out of domain
for arcsin (defined on [-1,1]), producing NaN on both CPU and GPU — but the
covariance matrix of NaN columns, fed to linalg.qr, behaves differently on
CPU (returns NaN Q) vs GPU (may raise a cuSOLVER error).
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1  = nn.Linear(8, 8)
bn   = nn.BatchNorm1d(8)
hs   = nn.Hardswish()
fc2  = nn.Linear(8, 4)

x = torch.randn(4, 8)

with torch.no_grad():
    x = fc1(x)
    x = torch.nn.functional.rrelu(x, lower=0.125, upper=0.333, training=False)
    x = bn(x)
    x = hs(x)
    x = fc2(x)
    x = x.requires_grad_(True)

with torch.enable_grad():
    jac = torch.autograd.grad(x.float_power(2).sum(), x, create_graph=True)[0]
    x2 = x + jac
    idx = x2.argsort(dim=1, descending=True)       # integer indices
    r   = idx.remainder_(3).float()                 # {0.0, 1.0, 2.0}
    print(f"arcsin input range: [{r.min():.1f}, {r.max():.1f}]")
    r_  = r.detach().clone()
    r_.arcsin_()                                    # arcsin(2.0) = NaN
    print(f"arcsin_ output: nan={torch.isnan(r_).any().item()}")
    cov = torch.cov(r_.T)
    print(f"cov: nan={torch.isnan(cov).any().item()}")

# Minimal 2-line version:
print("\nMinimal:")
v = torch.tensor([0.0, 1.0, 2.0])                  # 2.0 is out of arcsin domain
print(f"arcsin([0, 1, 2]) = {torch.arcsin(v)}")    # [0, π/2, nan]
