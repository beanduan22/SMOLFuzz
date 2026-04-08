"""
Bug S06 — INCONSISTENT: tan(ceil(sin*cos)) feeds divergent values into linalg.eigvalsh
Model: 0082
Trigger: add_noise mutation

Root cause: ceil(sin(y)*cos(y)) rounds the [-0.5,0.5] range to {-1, 0, 1}.
tan({-1, 0, 1}) = {-1.557, 0.0, 1.557} — well-defined. Then eigvalsh is
called on u @ u.T where u = conj_physical(tan(...)) and u is [4,8]. The product
u @ u.T is [4,4]. For identical integer-valued u on CPU vs GPU, eigvalsh
should agree, but the Gram matrix u @ u.T is rank-deficient (rank ≤ 3 for the
small-magnitude inputs), and cuSOLVER/LAPACK use different deflation strategies
for near-zero eigenvalues, producing reordered or sign-flipped results.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)

fc1 = nn.Linear(8, 8)

x = torch.randn(4, 8) + torch.randn(4, 8) * 0.05  # add_noise

def forward(f1, device):
    x_ = torch.randn(4, 8, device=device)
    x_.requires_grad_(True)
    with torch.enable_grad():
        y = f1(x_)
        z = torch.sin(y) * torch.cos(y)     # in [-0.5, 0.5]
        w = torch.ceil(z).int().float()      # {-1., 0., 1.}
        v = torch.tan(w)                     # {-1.557, 0., 1.557}
        u = torch.conj_physical(v)
        m = torch.linalg.eigvalsh(u @ u.T)  # eigenvalues of 4×4 Gram matrix
    return m

fc1_g = nn.Linear(8, 8).cuda(); fc1_g.load_state_dict(fc1.state_dict())

cpu_out = forward(fc1, 'cpu')
gpu_out = forward(fc1_g, 'cuda').cpu()

diff = (cpu_out - gpu_out).abs()
print(f"CPU eigenvalues: {cpu_out}")
print(f"GPU eigenvalues: {gpu_out}")
print(f"max |cpu - gpu|: {diff.max():.4e}")

# Minimal version: eigvalsh on a rank-deficient Gram matrix
print("\nMinimal:")
u = torch.tensor([[1., 0., -1., 1.],
                  [0., 1.,  0., 0.],
                  [1., 0., -1., 1.],
                  [1., 0., -1., 1.]])
g = u @ u.T
cpu_e = torch.linalg.eigvalsh(g)
gpu_e = torch.linalg.eigvalsh(g.cuda()).cpu()
print(f"eigvalsh diff: {(cpu_e - gpu_e).abs().max():.4e}")
