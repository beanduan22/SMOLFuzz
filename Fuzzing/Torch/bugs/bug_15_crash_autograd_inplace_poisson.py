"""
Bug #15 — CRASH: autograd version counter error from in-place op on poisson output
Cluster: 1 model (112)
Trigger: baseline

Root cause: torch.poisson is stochastic and non-differentiable. Passing its
output through a linear layer and then computing gradients via autograd triggers
a version-counter check on GPU. Additionally, register_hook on a non-leaf
scalar that was computed through a stochastic op may not be supported in all
PyTorch versions, causing a runtime error on GPU that is silently ignored on CPU.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

def forward(f1, f2, device):
    x = torch.randn(4, 8, device=device)
    with torch.enable_grad():
        x.requires_grad_(True)
        y = torch.poisson(torch.relu(x * 5))         # stochastic, non-differentiable
        z = f1(y)
        w = torch.special.log_softmax(z, dim=1)
        v = f2(w)
        u = torch.sin(v) + torch.cos(v)
        t = torch.triu(u)
        s = torch.nn.functional.binary_cross_entropy_with_logits(
            t, torch.zeros_like(t))
    s.register_hook(lambda grad: None)               # hook on stochastic-path scalar
    return s

# CPU
try:
    cpu_out = forward(fc1, fc2, 'cpu')
    print(f"CPU output: {cpu_out.item():.6f}, ok")
except Exception as e:
    print(f"CPU error: {type(e).__name__}: {e}")

# GPU
fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
fc2_g = nn.Linear(8, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
try:
    gpu_out = forward(fc1_g, fc2_g, 'cuda')
    print(f"GPU output: {gpu_out.item():.6f}, ok")
    print(f"Diff: {abs(cpu_out.item() - gpu_out.item()):.4e}  "
          f"(stochastic → not directly comparable)")
except Exception as e:
    print(f"GPU CRASH: {type(e).__name__}: {e}")
