"""
Bug #13 — CRASH: GPU crash via exp2 → argmax → Linear (model_0125)
Cluster: 1 model (125)  ← smallest crash (3 APIs)
Trigger: baseline

Root cause: exp2(fc1(x)) produces very large values. argmax(dim=1) reduces to
integer indices [0,7] and is cast to float, giving a [4,1] tensor. fc2 is
Linear(1, 8). On GPU, the forward pass through a Linear on this degenerate
(rank-1, integer-valued) input may trigger an internal cuBLAS assertion or
shape-validation error that does not occur on CPU.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(1, 8)

def forward(f1, f2, device):
    x = torch.randn(4, 8, device=device)
    x.requires_grad_(True)
    with torch.enable_grad():
        x = f1(x)
        x = torch.exp2(x)                                  # very large values
        argmax_x = torch.argmax(x, dim=1).unsqueeze(1).float()
        return f2(argmax_x)

# CPU
cpu_out = forward(fc1, fc2, 'cpu')
print(f"CPU output shape: {cpu_out.shape}, ok")

# GPU
fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
fc2_g = nn.Linear(1, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
try:
    gpu_out = forward(fc1_g, fc2_g, 'cuda')
    diff = (cpu_out - gpu_out.cpu()).abs()
    print(f"GPU output shape: {gpu_out.shape}, max diff: {diff.max():.4e}")
except Exception as e:
    print(f"GPU CRASH: {type(e).__name__}: {e}")
