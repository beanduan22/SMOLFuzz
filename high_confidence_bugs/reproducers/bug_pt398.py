#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 398
Bug Type: inconsistent
Detail  : output[0]: l2=9.4865e-03 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.functional.hardsigmoid, torch.nn.functional.hinge_embedding_loss, torch.fft.ifft2, torch.logcumsumexp
Mutation: scale_large

Run: python3 bug_pt398.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import ifft2

# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        # Gradient-tracking scope changes semantics of subsequent ops
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.fc1(x)
            z = F.hardsigmoid(y) * F.hinge_embedding_loss(y, torch.ones_like(y))
            w = ifft2(z)
            v = torch.logcumsumexp(w.real, 1)
            u = torch.sin(v) * torch.cos(v)
        return u

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

    # Embedded inputs (mutation: scale_large, already applied)
    x = torch.tensor([[702997.75, -278614.9375, -1230909.375, -217134.65625, -43574.2578125, 225041.203125, -941563.5, -334651.875], [-732716.75, 408498.78125, 208513.421875, -1657058.875, 572058.1875, 421530.0625, 1338393.25, -162130.703125], [1370748.625, 1195530.875, -487297.625, -751929.0625, 285215.78125, 122924.59375, -93846.0703125, -1419654.875], [1378770.25, 362684.28125, -530503.875, 643936.0, 1771128.625, -514009.21875, -1075299.75, 433106.40625]], dtype=torch.float32)
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
