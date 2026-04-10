#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 396
Bug Type: nan
Detail  : output[0]: asymmetric non-finite values asym_nan=0 asym_inf=0 nan_vs_inf=1 shape=[]
APIs    : torch.nn.Linear, torch.nn.LeakyReLU, torch.distributions.transforms.AbsTransform, torch.ormqr
Mutation: uniform

Run: python3 bug_pt396.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.transforms as transforms

# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.leaky_relu = nn.LeakyReLU()
        self.abs_transform = transforms.AbsTransform()
        self.ormqr_linear = nn.Linear(16, 8)
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(1)  # unused due to shape constraints
        self.sym_eig_linear = nn.Linear(8, 4)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = self.leaky_relu(y)
            w = self.ormqr_linear(z)
            v = torch.arccosh(w + 2)  # ensure domain of arccosh
            u = self.sym_eig_linear(v)
            log_det = torch.logdet(u)
            reciprocal_u = u.reciprocal_()
        return log_det, reciprocal_u

def make_inputs():
    return [torch.randn(4, 8)]

# ── Reproducer ─────────────────────────────────────────────────
def run():
    torch.manual_seed(42)
    cpu_model = Model()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    gpu_model = Model().cuda()
    gpu_model.load_state_dict(cpu_model.state_dict())
    cpu_model.eval()
    gpu_model.eval()

    # Embedded inputs (mutation: uniform, already applied)
    x = torch.tensor([[0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386], [0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386], [0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386], [0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386, 0.4033372700214386]], dtype=torch.float32)
    inputs = [x]

    with torch.no_grad():
        cpu_out = cpu_model(*inputs)
        gpu_out = gpu_model(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs])

    if isinstance(cpu_out, torch.Tensor):
        cpu_outs, gpu_outs = [cpu_out], [gpu_out.cpu()]
    else:
        cpu_outs = [o for o in cpu_out if isinstance(o, torch.Tensor)]
        gpu_outs = [o.cpu() for o in gpu_out if isinstance(o, torch.Tensor)]

    found = False
    for i, (c, g) in enumerate(zip(cpu_outs, gpu_outs)):
        g = g.cpu()
        if c.shape != g.shape:
            print(f"output[{i}]: shape mismatch cpu={c.shape} gpu={g.shape}")
            found = True
            continue
        cf, gf = c.float(), g.float()
        fin = ~(torch.isnan(cf) | torch.isinf(cf) | torch.isnan(gf) | torch.isinf(gf))
        asym = ((torch.isnan(cf) & ~torch.isnan(gf)) | (torch.isnan(gf) & ~torch.isnan(cf)) |
                (torch.isinf(cf) & ~torch.isinf(gf)) | (torch.isinf(gf) & ~torch.isinf(cf)) |
                (torch.isnan(cf) & torch.isinf(gf)) | (torch.isinf(cf) & torch.isnan(gf)))
        if asym.any():
            print(f"output[{i}]: ASYMMETRIC NaN/Inf — cpu_nan={torch.isnan(cf).sum().item()} gpu_nan={torch.isnan(gf).sum().item()} "
                  f"cpu_inf={torch.isinf(cf).sum().item()} gpu_inf={torch.isinf(gf).sum().item()} asym={asym.sum().item()}")
            found = True
        elif fin.any():
            l2 = (cf[fin] - gf[fin]).pow(2).sum().sqrt().item()
            if l2 > 1e-3:
                print(f"output[{i}]: L2={l2:.4e}  shape={list(c.shape)}")
                found = True

    if not found:
        print("No inconsistency detected (may be hardware/driver dependent)")
    else:
        print("BUG CONFIRMED: CPU and GPU produce different results for the same model and input")

if __name__ == "__main__":
    run()
