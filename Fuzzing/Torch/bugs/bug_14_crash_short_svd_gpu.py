"""
Bug #14 — CRASH: GPU crash via .short() truncation → matrix_exp → svd chain (model_0099)
Cluster: 1 model (99)
Trigger: baseline

Root cause: The chain converts float tensors to int16 via .short() then back to
float. This quantises values to integers. After several ops, matrix_exp is
applied to a diag_embed of scalar means — diag_embed creates a [4,1,1] matrix
whose exponential is trivial. Then torch.svd is called on a [4,1,1] tensor.
On GPU, cuSOLVER SVD may disagree with CPU LAPACK on sign conventions for the
U/V singular vectors, or the very small matrix may hit a degenerate codepath.
"""
import torch
import torch.nn as nn
import math

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
bn  = nn.BatchNorm1d(8)
fc2 = nn.Linear(8, 8)

def forward(f1, b, f2, device):
    x = torch.randn(4, 8, device=device)
    x.requires_grad_(True)
    with torch.enable_grad():
        x = f1(x)
        x = torch.nn.functional.huber_loss(x, torch.zeros_like(x), reduction='none')
        x = x.flip(0)
        x = x.short().float()                        # int16 quantisation
        x = b(x)
        x = f2(x)
        x = (1 + torch.erf(x / math.sqrt(2))) / 2
        x = x.remainder(1.0)
        x = torch.tanh(x)
        x = torch.clamp(x, -0.99, 0.99)
        x = torch.sin(x)
        _, x = torch.std_mean(x, dim=1, keepdim=True)
        x = torch.matrix_exp(torch.diag_embed(x))   # [4,1,1] matrix
        u, s, vh = torch.svd(x)
    return s.sum(dim=1, keepdim=True)

fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
bn_g  = nn.BatchNorm1d(8).cuda(); bn_g.load_state_dict(bn.state_dict())
fc2_g = nn.Linear(8, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())

cpu_out = forward(fc1, bn, fc2, 'cpu')
print(f"CPU output: {cpu_out.flatten()}, ok")

try:
    gpu_out = forward(fc1_g, bn_g, fc2_g, 'cuda')
    diff = (cpu_out - gpu_out.cpu()).abs()
    print(f"GPU output: {gpu_out.flatten().cpu()}, max diff: {diff.max():.4e}")
except Exception as e:
    print(f"GPU CRASH: {type(e).__name__}: {e}")
