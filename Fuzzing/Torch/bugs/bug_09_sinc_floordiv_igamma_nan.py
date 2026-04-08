"""
Bug #9 — NAN: masked input feeds NaN through sinc → floor_divide → igamma chain
Cluster: 2 models (59, 202)
Representative: model_0059
Trigger: mask mutation

Root cause: exponential_() draws from Exp(1), producing strictly positive values.
sinc(x) = sin(πx)/(πx) is well-defined everywhere but floor_divide_(0.5)
converts continuous values to integers (0, 2, 4, …), then igamma(a, x) with
a=1 and x=0 (from masked zeros) evaluates the lower incomplete gamma function
at its boundary, which can produce NaN on GPU. CPU and GPU differ in how they
handle domain-edge inputs to igamma.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

ln  = nn.LayerNorm(8)
fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

x = torch.randn(4, 8)
# mask mutation: zero some elements
mask = torch.bernoulli(torch.full((4, 8), 0.5)).bool()
x[mask] = 0.0

with torch.no_grad():
    x = ln(x)
    x = torch.sin(x)
    x = fc1(x)
    x = x.exponential_()               # all positive now
    x = fc2(x)
    x = torch.sinc(x)                  # sinc of the result
    x = x.floor_divide(torch.tensor(0.5))  # collapses to integers including 0
    x = x.resolve_conj()
    out = x.igamma(torch.ones_like(x)) # igamma at boundary (x=0)

print(f"floor_divide result range: [{x.min():.2f}, {x.max():.2f}]")
print(f"zero elements: {(x == 0).sum().item()}")
print(f"igamma output: nan={torch.isnan(out).any().item()}, inf={torch.isinf(out).any().item()}")

# Minimal version: igamma at x=0
print("\nMinimal:")
print(f"igamma(1, 0) = {torch.tensor(0.0).igamma(torch.tensor(1.0))}")
