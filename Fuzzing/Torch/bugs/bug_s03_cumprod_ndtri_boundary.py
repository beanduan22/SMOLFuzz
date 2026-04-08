"""
Bug S03 — NAN: cumprod zeros out rows, feeding 0/1 boundary into ndtri (model_0044)
Trigger: add_noise mutation

Root cause: gt(w, 0.5) produces a Boolean mask cast to float: {0.0, 1.0}.
cumprod of a sequence containing 0.0 gives 0.0 for all subsequent elements.
After matmul, linalg.matrix_rank returns a scalar integer (rank). ndtri
(the quantile function / inverse-normal CDF) of an integer: ndtri(0) = -inf,
ndtri(1) = +inf, ndtri(k≥2) = NaN. These boundary values propagate as NaN.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

x = torch.randn(4, 8) + torch.randn(4, 8) * 0.05   # add_noise mutation

with torch.enable_grad():
    x = x.requires_grad_(True)
    y = fc1(x)
    z = torch.nn.functional.selu(y)
    w = torch.nn.functional.leaky_relu(z)
    v = torch.gt(w, 0.5).float()             # {0.0, 1.0}
    s = torch.cumprod(v, dim=0)              # zeros propagate forward
    r = torch.matmul(s.t(), w)              # [8, 8] but rank-deficient
    q = torch.linalg.matrix_rank(r)         # integer scalar (e.g. 2)
    print(f"matrix_rank = {q.item()}")
    p = torch.special.ndtri(q.float())      # ndtri of integer ≥ 1

print(f"ndtri({q.item()}) = {p.item()}")    # likely NaN

# Minimal version:
print("\nMinimal:")
for v in [0.0, 0.5, 1.0, 2.0]:
    print(f"ndtri({v}) = {torch.special.ndtri(torch.tensor(v)).item():.4f}")
# ndtri(0)=−∞, ndtri(0.5)=0, ndtri(1)=+∞, ndtri(2)=NaN
