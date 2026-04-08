"""
Bug S07 — NAN: log(ELU(x) + eps) produces NaN when ELU output ≈ -eps (model_0256)
Trigger: uniform mutation

Root cause: ELU(x) = x if x>0, else α*(e^x - 1). For x slightly negative,
ELU(x) ∈ (-1, 0). The guard eps=1e-6 ensures log(ELU + eps) > log(0).
However, with uniform inputs in [-1,1], some x values produce ELU(x) ≈ -1e-6
(i.e. ELU+eps ≈ 0). CPU and GPU compute e^x differently at the ULP level for
x near 0, so CPU may give ELU+eps = 1e-8 (log finite) while GPU gives
ELU+eps = -3e-9 (log of negative = NaN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

fc1 = nn.Linear(8, 16)
fc2 = nn.Linear(16, 8)
softmax = nn.Softmin(dim=1)

def forward(f1, f2, sm, device):
    x = torch.zeros(4, 8, device=device).uniform_(-1.0, 1.0)  # uniform mutation
    x = x.requires_grad_(True)
    with torch.enable_grad():
        y = F.elu(x)
        z = torch.log(y + 1e-6)         # guard: but ELU can be ≈ -1e-6
        w = f1(z)
        v = torch.sin(w) * torch.cos(w)
        u = f2(v)
        t = torch.true_divide(u, u + 1e-6)
        s = sm(t)
    return s

fc1_g = nn.Linear(8, 16).cuda(); fc1_g.load_state_dict(fc1.state_dict())
fc2_g = nn.Linear(16, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
sm_g  = nn.Softmin(dim=1).cuda()

cpu_out = forward(fc1, fc2, softmax, 'cpu')
gpu_out = forward(fc1_g, fc2_g, sm_g, 'cuda').cpu()

print(f"CPU: nan={torch.isnan(cpu_out).any().item()}, inf={torch.isinf(cpu_out).any().item()}")
print(f"GPU: nan={torch.isnan(gpu_out).any().item()}, inf={torch.isinf(gpu_out).any().item()}")

# Minimal version: ELU + eps near boundary
print("\nMinimal: ELU near -eps")
for x in [-1e-5, -1e-6, -1e-7, 0.0]:
    elu_val = F.elu(torch.tensor(x))
    log_val = torch.log(elu_val + 1e-6)
    print(f"  x={x:.1e}: ELU={elu_val.item():.4e}, ELU+eps={elu_val.item()+1e-6:.4e}, log={log_val.item():.4f}")
