"""
Bug #5 — INCONSISTENT: torch.sin/cos CPU/GPU divergence after add_noise mutation
Cluster: 6 models (35, 82, 116, 166, 181, 206)
Representative: model_0166
Trigger: add_noise mutation (small Gaussian noise added to inputs)

Root cause: Adding noise pushes floating-point values across rounding boundaries
where CPU and GPU sin/cos approximations round in opposite directions. The
subsequent tanh and gradient computation (autograd.grad) propagate the error.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

def forward(x):
    x = x.requires_grad_(True)
    y = torch.sin(fc1(x))
    z = torch.cos(y)
    w = torch.tanh(z)
    v = fc2(w)
    ones = torch.ones_like(v)
    loss = torch.nn.functional.triplet_margin_loss(v, ones, -ones)
    return torch.autograd.grad(loss.sum(), x)[0]

x_base = torch.randn(4, 8)
noise  = torch.randn(4, 8) * 0.05   # add_noise mutation
x      = x_base + noise

fc1_g = nn.Linear(8, 8).cuda()
fc2_g = nn.Linear(8, 8).cuda()
fc1_g.load_state_dict(fc1.state_dict())
fc2_g.load_state_dict(fc2.state_dict())

def forward_gpu(x):
    x = x.requires_grad_(True)
    y = torch.sin(fc1_g(x))
    z = torch.cos(y)
    w = torch.tanh(z)
    v = fc2_g(w)
    ones = torch.ones_like(v)
    loss = torch.nn.functional.triplet_margin_loss(v, ones, -ones)
    return torch.autograd.grad(loss.sum(), x)[0]

cpu_out = forward(x)
gpu_out = forward_gpu(x.cuda()).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
print(f"oracle tol=1e-06, divergence detected: {diff.max() > 1e-6}")
