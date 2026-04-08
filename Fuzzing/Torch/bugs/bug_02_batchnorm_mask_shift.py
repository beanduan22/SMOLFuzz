"""
Bug #2 — INCONSISTENT: BatchNorm1d CPU/GPU stat divergence under masked input
Cluster: 12 models (91, 134, 147, 149, 161, 168, 178, 213, 215, 230, 244, 255)
Representative: model_0091
Trigger: mask mutation (fraction of inputs zeroed)

Root cause: The mask mutation zeros a subset of batch elements, shifting the
empirical mean/variance seen by BatchNorm. CPU and GPU compute the per-batch
statistics using different parallel-reduction orderings, producing ULP-level
differences in the normalised output. Any nonlinear op after BN (here sin)
amplifies the discrepancy.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 16)
bn  = nn.BatchNorm1d(16)
fc2 = nn.Linear(16, 8)

def forward(x):
    x = fc1(x)
    x = bn(x)
    x = torch.nn.functional.leaky_relu(x)
    x = -x
    x = torch.special.log_softmax(x, dim=1)
    x = torch.sin(x)
    x = torch.clamp(x, -1.0, 1.0)
    return fc2(x)

x = torch.randn(4, 8)
x[2, :] = 0.0       # mask mutation: zero out some rows
x[3, :] = 0.0

cpu_model = nn.Sequential().eval()   # use module-level BN so we can copy weights

# CPU pass
fc1_c, bn_c, fc2_c = fc1, bn, fc2
cpu_out = forward(x)

# GPU pass — clone all parameters to GPU
fc1_g = nn.Linear(8, 16).cuda()
bn_g  = nn.BatchNorm1d(16).cuda()
fc2_g = nn.Linear(16, 8).cuda()
fc1_g.load_state_dict(fc1.state_dict())
bn_g.load_state_dict(bn.state_dict())
fc2_g.load_state_dict(fc2.state_dict())

def forward_gpu(x):
    x = fc1_g(x)
    x = bn_g(x)
    x = torch.nn.functional.leaky_relu(x)
    x = -x
    x = torch.special.log_softmax(x, dim=1)
    x = torch.sin(x)
    x = torch.clamp(x, -1.0, 1.0)
    return fc2_g(x)

gpu_out = forward_gpu(x.cuda()).cpu()

diff = (cpu_out - gpu_out).abs()
print(f"max |cpu - gpu|: {diff.max():.4e}")
print(f"oracle tol=1e-06, divergence detected: {diff.max() > 1e-6}")
