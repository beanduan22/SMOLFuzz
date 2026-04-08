"""
Bug S04 — CRASH: NaN produced by sin chain causes assert_close / comparison failure (model_0062)
Trigger: baseline

Root cause: chain_matmul(w, w.T, w[:,:4]) requires specific shape compatibility;
if w is [4,8] then w[:,:4] is [4,4] and w.T is [8,4], so chain is [4,8]×[8,4]×[4,4]=[4,4].
torch.inverse of a nearly-singular matrix (sin*cos values → small determinant)
returns a matrix that may contain inf/NaN. Then autograd.grad through the second
sin*cos block accumulates NaN gradients. When the oracle compares CPU and GPU
outputs using assert_close, NaN ≠ NaN by IEEE 754, so the comparison fails and
is misclassified as a "crash" rather than NaN.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)
fc2 = nn.Linear(8, 8)

def forward(f1, f2, device):
    x = torch.randn(4, 8, device=device)
    with torch.enable_grad():
        y = f1(x)
        z = torch.sin(y) * torch.cos(y)
        w = z + torch.ones_like(z)
        # chain_matmul: [4,8] × [8,4] × [4,4] → [4,4]
        v = torch.chain_matmul(w, w.T, w[:, :4])
        u = torch.inverse(v)                   # may produce nan/inf for near-singular v

    x.requires_grad_(True)
    with torch.enable_grad():
        a = f2(x)
        b = torch.sin(a) * torch.cos(a)
        c = b + torch.ones_like(b)
        g = torch.autograd.grad(c.sum(), a)[0]
    return torch.mean(g)

cpu_out = forward(fc1, fc2, 'cpu')
print(f"CPU output: {cpu_out.item()}")
print(f"CPU output is nan: {torch.isnan(cpu_out).item()}")

fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())
fc2_g = nn.Linear(8, 8).cuda(); fc2_g.load_state_dict(fc2.state_dict())
gpu_out = forward(fc1_g, fc2_g, 'cuda')
print(f"GPU output: {gpu_out.item()}")
print(f"GPU output is nan: {torch.isnan(gpu_out).item()}")

# The "crash": NaN != NaN under assert_close
print(f"\ntorch.testing.assert_close on (nan, nan):")
try:
    torch.testing.assert_close(
        torch.tensor(float('nan')),
        torch.tensor(float('nan'))
    )
    print("passed (equal_nan=True by default in new PyTorch)")
except AssertionError as e:
    print(f"FAILS: {e}")
