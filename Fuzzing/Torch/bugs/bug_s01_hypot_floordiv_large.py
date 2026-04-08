"""
Bug S01 — INCONSISTENT: torch.hypot + floor_divide integer truncation at large scale
Model: 0036
Trigger: scale_large

Root cause: hypot(z, z) = z * sqrt(2) on a ReLU output. For large inputs,
the result is large. floor_divide(z, 2) performs integer division. The
z.short() cast truncates float to int16, clamping values outside [-32768, 32767].
CPU and GPU differ in how they handle overflow of the short cast for large floats,
and the subsequent integer-valued tensor operations diverge.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 16)

x = torch.randn(4, 8) * 100.0     # scale_large

with torch.no_grad():
    y  = fc1(x)
    z  = torch.relu(y)             # all positive
    hz = torch.hypot(z, z)         # z * sqrt(2), large
    zs = hz.short()                # int16 truncation — overflows if hz > 32767
    zf = zs.float()
    zm = torch.max(hz, dim=1)[0]   # scalar per row

print(f"hypot range: [{hz.min():.1f}, {hz.max():.1f}]")
print(f"short overflow (> 32767): {(hz > 32767).sum().item()} elements")
print(f"after short→float range: [{zf.min():.1f}, {zf.max():.1f}]")

# Minimal version: short cast overflow
big = torch.tensor([40000.0, -40000.0, 100.0])
print(f"\nbig.short().float() = {big.short().float()}")  # wrapped/clamped values
