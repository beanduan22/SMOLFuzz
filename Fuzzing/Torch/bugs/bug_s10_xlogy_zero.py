"""
Bug S10 — NAN: torch.xlogy(x, x) as x → 0 gives 0*log(0) = NaN (model_0297)
Trigger: baseline

Root cause: xlogy(x, y) = x * log(y). When x=0 and y=0, this should be 0
(by the convention lim_{x→0} x*log(x) = 0), but PyTorch returns NaN for
xlogy(0, 0) because it evaluates 0 * log(0) = 0 * (-inf) = NaN.

The model applies hardswish (which can output 0), then BatchNorm, then log1p,
then xlogy(x, x). If any element is exactly 0 after these transforms, xlogy
produces NaN. CPU and GPU agree on the NaN, but the oracle flags it as a bug.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)
bn  = nn.BatchNorm1d(8)

x = torch.randn(4, 8)

with torch.enable_grad():
    x = x.requires_grad_(True)
    x_ = fc1(x)
    x_ = torch.nn.functional.hardswish(x_)   # can output exactly 0
    x_ = bn(x_)
    x_ = fc2(x_)
    x_ = torch.log1p(x_)
    out = torch.xlogy(x_, x_)                 # xlogy(0, 0) = NaN if any element = 0

print(f"hardswish zeros: {(torch.nn.functional.hardswish(fc1(x).detach()) == 0).sum().item()}")
print(f"xlogy output: nan={torch.isnan(out).any().item()}")

# Minimal 3-line version:
print("\nMinimal: xlogy at x=0")
v = torch.tensor([0.0, 0.5, 1.0, 2.0])
print(f"xlogy(v, v) = {torch.xlogy(v, v)}")
# Expected: [nan, -0.3466, 0.0, 1.3863]
# NaN because xlogy(0, 0) = 0 * log(0) = 0 * -inf = NaN

# The mathematically correct value:
print(f"lim_{{x→0}} x*log(x) = 0, but torch gives: {torch.xlogy(torch.tensor(0.0), torch.tensor(0.0))}")
