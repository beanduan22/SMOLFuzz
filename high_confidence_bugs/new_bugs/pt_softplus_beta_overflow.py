"""Bug 51: torch.nn.functional.softplus with large beta returns inf instead of correct value.
Source: PyTorch GitHub issue #171249
"""
import torch
import torch.nn.functional as F

x = torch.tensor([0.5, 1.0, 2.0])
cpu = F.softplus(x, beta=1e30, threshold=float('inf'))
gpu = F.softplus(x.cuda(), beta=1e30, threshold=float('inf')).cpu()

print("CPU:", cpu.tolist())       # BUG: [inf, inf, inf]
print("GPU:", gpu.tolist())       # BUG: [inf, inf, inf]
print("Expected: [0.5, 1.0, 2.0]")
assert torch.isinf(cpu).any() and torch.isinf(gpu).any(), "bug not reproduced"
print("BUG CONFIRMED: softplus overflow with large beta")
