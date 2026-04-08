"""
Bug S05 — NAN: erfinv(arcsin(sin*cos)) overflows near ±1 (model_0070)
Trigger: scale_small mutation (GPU-only NaN)

Root cause: sin(y)*cos(y) = sin(2y)/2, bounded in [-0.5, 0.5].
arcsin of a value in [-0.5, 0.5] gives a value in [-π/6, π/6] ≈ [-0.524, 0.524].
After summing across 16 outputs and taking median, values near ±1 are possible.
erfinv is extremely sensitive near ±1: erfinv(1-ε) ~ sqrt(-log(ε)).
On GPU, small ULP differences in arcsin push the argument just above 1,
producing NaN on GPU but not CPU (or vice versa).
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 16)
fc2 = nn.Linear(1, 8)

x = torch.randn(4, 8) * 0.01    # scale_small: near-zero regime

with torch.enable_grad():
    x = x.requires_grad_(True)
    y  = fc1(x)
    z  = torch.sin(y) * torch.cos(y)     # in [-0.5, 0.5]
    w  = torch.arcsin(z)                  # in [-π/6, π/6]
    v  = torch.sum(w, dim=1)             # sum of 16 values, ~[-8.4, 8.4]
    u  = torch.median(v.unsqueeze(0))    # scalar median
    t  = torch.erfinv(u)                 # sensitive near ±1
    s  = torch.fix(t)
    r  = torch.i0(s).unsqueeze(0)
    q  = fc2(r.T)

print(f"arcsin range: [{w.min():.4f}, {w.max():.4f}]")
print(f"median value: {u.item():.6f}")
print(f"erfinv input (clamped perspective): {'in-domain' if u.abs() < 1 else 'OUT OF DOMAIN'}")
print(f"erfinv result: {t.item()}")
print(f"is nan: {torch.isnan(t).item()}")

# Minimal version: erfinv sensitivity near ±1
print("\nMinimal: erfinv near boundary")
for v in [0.9, 0.99, 0.999, 1.0, 1.001]:
    print(f"  erfinv({v}) = {torch.erfinv(torch.tensor(v)).item():.4f}")
