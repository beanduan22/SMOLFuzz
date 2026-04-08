"""
Bug S09 — NAN: sin output clipped and cast to half() loses precision → prod NaN (model_0289)
Trigger: scale_large mutation

Root cause: sin(large_x) produces values scattered in [-1,1]. clip_(0,1) keeps
the positive part. Casting to half (float16) is fine for values in [0,1], but
if any upstream computation produced inf (e.g. from sigmoid of large fc2 output),
clip_ clamps to 1.0 which is representable in half — but if the pre-clip value
was NaN, clip_ preserves NaN. Then prod() of a row containing NaN is NaN.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

linear1  = nn.Linear(8, 8)
sigmoid  = nn.Sigmoid()
linear2  = nn.Linear(8, 8)
kldivloss = nn.KLDivLoss(reduction='batchmean')

x = torch.randn(4, 8) * 100.0   # scale_large

with torch.enable_grad():
    x = x.requires_grad_(True)
    x_ = linear1(x)
    x_ = sigmoid(x_)
    x_ = linear2(x_)
    x_ = torch.sin(x_)
    x_ = x_.clip_(0.0, 1.0)
    x_h = x_.half()             # float16 cast

print(f"pre-clip range:  [{x_.min():.4f}, {x_.max():.4f}]")
print(f"half() nan: {torch.isnan(x_h).any().item()}, inf: {torch.isinf(x_h).any().item()}")

prod_out = torch.prod(x_h.float(), dim=1).unsqueeze(1)
print(f"prod output: {prod_out.flatten()}")
print(f"prod nan: {torch.isnan(prod_out).any().item()}")

# Minimal version: clip_ does not help NaN
print("\nMinimal: clip_ on NaN")
v = torch.tensor([float('nan'), 0.5, 1.0, -0.5])
v.clip_(0.0, 1.0)
print(f"clip_(nan, 0.5, 1.0, -0.5) = {v}")  # NaN survives clip_
print(f"nan survives clip_: {torch.isnan(v[0]).item()}")
print(f"prod with NaN: {torch.prod(v).item()}")
