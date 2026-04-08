"""
Bug #7 — NAN: torch.special.erfinv receives out-of-domain input from masked sum
Cluster: 2 models (79, 88)
Representative: model_0079
Trigger: mask mutation

Root cause: erfinv(x) is defined only for x ∈ (-1, 1). The mask mutation
zeros some elements; relu + masked_fill keeps values in [0.5, ∞). After
summing 8 such values and broadcasting, the input to erfinv easily exceeds 1.
CPU returns ±inf; GPU may return NaN — hence CPU/GPU inconsistency.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)
bn  = nn.BatchNorm1d(8)

x = torch.randn(4, 8)

with torch.no_grad():
    x = fc1(x)
    x = torch.nn.functional.relu(x)
    x = bn(x)
    x = fc2(x)
    x = torch.sin(x)
    x = x.masked_fill(x < 0.5, 0.0)    # mask: keep only values ≥ 0.5
    x = x.repeat(1, 2)[:, :8]
    x = torch.sum(x, dim=1, keepdim=True).expand(-1, 8)  # row sums, easily > 1

print(f"erfinv input range: [{x.min():.4f}, {x.max():.4f}]")
print(f"values outside (-1,1): {(x.abs() > 1).sum().item()}")

out = torch.special.erfinv(x)
print(f"erfinv output: contains nan={torch.isnan(out).any().item()}, "
      f"inf={torch.isinf(out).any().item()}")

# Minimal 2-line version:
print("\nMinimal:")
v = torch.tensor([0.0, 1.5, 2.0, -1.5])   # some values out of (-1,1)
print(f"erfinv({v}) = {torch.special.erfinv(v)}")  # expect nan or ±inf
