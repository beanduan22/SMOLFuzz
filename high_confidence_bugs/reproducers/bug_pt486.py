#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 486
Bug Type: inconsistent
Detail  : output[0]: l2=3.0295e-03 > threshold=1e-03 finite_elements=32 shape=[4, 8]; output[1]: l2=1.7158e-03 > threshold=1e-03 f
APIs    : torch.nn.Linear, torch.sin, torch.cos, torch.nn.Tanhshrink
Mutation: scale_large

Run: python3 bug_pt486.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.tanhshrink = nn.Tanhshrink()

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            x = self.fc1(x)
            x = torch.sin(x) * torch.cos(x)
            x = self.tanhshrink(x)
            x = self.fc2(x)
            y = x.var(dim=0)
            z = torch.erf(x)
            jac = torch.autograd.grad(y.sum(), x, allow_unused=True)[0]
        return x, jac, z

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
    x = torch.tensor([[294037.3125, 414175.40625, -65990.6328125, -923997.625, -1039047.125, -640162.6875, -727251.5625, -100772.796875], [265938.4375, -287175.125, -151044.046875, 382098.4375, 99202.1796875, -348049.78125, 758885.625, 9890.4150390625], [-158432.640625, -224531.609375, 468144.875, -23151.515625, -356240.0625, -227377.234375, 99491.8203125, -332768.34375], [-284873.96875, 1061222.0, 820382.3125, -850360.4375, 202728.375, -132002.28125, -755424.5625, -826808.875]], dtype=torch.float32)
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
