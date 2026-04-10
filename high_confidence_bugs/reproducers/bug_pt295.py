#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 295
Bug Type: inconsistent
Detail  : output[0]: l2=2.8284e+00 > threshold=1e-03 finite_elements=16 shape=[4, 4]
APIs    : torch.nn.Linear, torch.sin, torch.nn.Softshrink, torch.nn.BatchNorm1d
Mutation: uniform

Run: python3 bug_pt295.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 4)
        self.softshrink = nn.Softshrink(lambd=0.5)
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            x = self.linear1(x)
            x = torch.sin(x)
            x = self.softshrink(x)
            x = self.linear2(x)
            self.train()
            x = self.bn(x)
            x = self.bn(x)  # Reuse BN to demonstrate mode change
            self.eval()
        x = x.remainder(1.0)
        return x

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
    x = torch.tensor([[1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555], [1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555], [1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555], [1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555, 1.4817914962768555]], dtype=torch.float32)
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
