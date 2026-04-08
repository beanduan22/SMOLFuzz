"""
Bug #11 — CRASH: GPU crash in sin/cos → mm → renorm chain (model_0076)
Cluster: 1 model (76)
Trigger: baseline

Root cause: The forward pass builds a [4,4] matrix via mm(z, w.T) where z and w
are [4,8]. Then triu_indices is constructed on the target device, and renorm is
applied. On GPU, renorm with p=2, dim=0 over a matrix that may have near-zero
column norms can produce a CUDA illegal memory access or a cuBLAS error,
depending on driver version. The model runs fine on CPU.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

def forward(model_fc1, model_fc2, device):
    x = torch.randn(4, 8, device=device)
    y = model_fc1(x)
    z = torch.sin(y) * torch.cos(y)
    w = torch.log1p(z.abs())
    v = torch.mm(z, w.t())                            # [4,4]
    u = torch.triu_indices(8, 8, device=device)
    t = torch.renorm(w, p=2, dim=0, maxnorm=10.0)     # potential GPU fault
    s = torch.sin(t)
    return model_fc2(s.detach())

# CPU — should succeed
cpu_out = forward(fc1, fc2, 'cpu')
print(f"CPU output shape: {cpu_out.shape}, ok")

# GPU — may crash
fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
fc2_g = nn.Linear(8, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
try:
    gpu_out = forward(fc1_g, fc2_g, 'cuda')
    diff = (cpu_out - gpu_out.cpu()).abs()
    print(f"GPU output shape: {gpu_out.shape}, max diff: {diff.max():.4e}")
except Exception as e:
    print(f"GPU CRASH: {type(e).__name__}: {e}")
