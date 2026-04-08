"""
Bug #10 — INCONSISTENT: SELU/Softshrink CPU/GPU divergence with uniform input
Cluster: 2 models (123, 197)
Representative: model_0123
Trigger: uniform mutation

Root cause: SELU's negative-saturation region uses e^x scaled by a fixed
constant (α≈1.6733, λ≈1.0507). With uniform inputs, more batch elements
land exactly in the soft transition zone of Softshrink (within λ of zero),
and in SELU's exponential region. CPU and GPU evaluate the exponential
differently at the boundary, producing a small but consistent divergence
in the cumsum output.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

linear1    = nn.Linear(8, 8)
softshrink = nn.Softshrink(lambd=0.5)
softmax    = nn.Softmax(dim=1)
selu       = nn.SELU()

def forward(x):
    x = linear1(x)
    x = softshrink(x)
    x = softmax(x)
    x = selu(x)
    x = torch.cumsum(x, dim=1)
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

l_g  = nn.Linear(8, 8).cuda();       l_g.load_state_dict(linear1.state_dict())
ss_g = nn.Softshrink(lambd=0.5).cuda()
sm_g = nn.Softmax(dim=1).cuda()
se_g = nn.SELU().cuda()

def forward_gpu(x):
    x = l_g(x)
    x = ss_g(x)
    x = sm_g(x)
    x = se_g(x)
    x = torch.cumsum(x, dim=1)
    return torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

x = torch.zeros(4, 8).uniform_(-1.0, 1.0)   # uniform mutation

cpu_out = forward(x)
gpu_out = forward_gpu(x.cuda()).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
print(f"oracle tol=1e-06, divergence detected: {diff.max() > 1e-6}")
