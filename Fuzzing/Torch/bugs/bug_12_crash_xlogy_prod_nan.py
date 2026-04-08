"""
Bug #12 — NAN/CRASH: torch.special.xlogy(x, x) when x=0 via prod of sin values
Cluster: 1 model (53)
Trigger: baseline

Root cause: torch.prod(sin(x), dim=1) multiplies 8 sin values together.
If any element is exactly 0, the product is 0. Then xlogy(0, 0) computes
0 * log(0). Mathematically this is 0 (by L'Hôpital), but floating-point
gives 0 * -inf = NaN. CPU and GPU may handle this differently.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1  = nn.Linear(8, 8)
drop = nn.Dropout(p=0.3)
rrelu = nn.RReLU()
fc2  = nn.Linear(1, 8)

x = torch.randn(4, 8)

with torch.no_grad():
    x = fc1(x)
    x = torch.where(x > 0, x, torch.zeros_like(x))   # zero out negatives
    x = torch.sin(x)                                   # sin(0) = 0 is possible
    x = drop(x)                                        # dropout may add more zeros
    x = rrelu(x)
    prod = torch.prod(x, dim=1, keepdim=True)          # product of 8 values → often 0
    xlogy_out = torch.special.xlogy(prod, prod)        # 0 * log(0) → NaN

print(f"prod zeros: {(prod == 0).sum().item()} / {prod.numel()}")
print(f"xlogy(prod, prod) contains NaN: {torch.isnan(xlogy_out).any().item()}")

# Minimal 2-line version:
print("\nMinimal:")
z = torch.tensor([0.0, 1.0, 2.0])
print(f"xlogy(z, z) = {torch.special.xlogy(z, z)}")  # [nan, 0, ~1.386]
