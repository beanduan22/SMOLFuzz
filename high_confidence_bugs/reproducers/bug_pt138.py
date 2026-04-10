#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 138
Bug Type: inconsistent
Detail  : output[0]: l2=2.0447e+00 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.sin, torch.cos, torch.tanh
Mutation: scale_small

Run: python3 bug_pt138.py
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
        self.identity = nn.Identity()

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.sin(x)
            z = self.fc1(y)
            w = torch.cos(z)
            v = self.fc2(w)
            u = torch.tanh(v)
            jacobian = torch.autograd.functional.jacobian(lambda t: torch.sin(self.fc1(t)), x)
            loss = nn.NLLLoss()(u, torch.empty_like(u).argmin(dim=1))
            with torch.no_grad():
                cholesky_solve_result = torch.cholesky_solve(torch.eye(8), torch.eye(8))
        return u + jacobian.sum() + loss + cholesky_solve_result.sum()

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

    # Embedded inputs (mutation: scale_small, already applied)
    x = torch.tensor([[0.01840631663799286, 0.04105927795171738, -0.029126323759555817, -0.026505636051297188, 0.010902647860348225, 0.05237780138850212, 0.017348673194646835, -0.046151451766490936], [-0.05771376192569733, 0.05008180812001228, -0.0659584328532219, -0.08147940784692764, -0.02681438997387886, 0.01972654089331627, -0.012681521475315094, -0.029163626953959465], [0.011069455184042454, -0.10323746502399445, 0.03578886017203331, -0.019567597657442093, 0.011541363783180714, 0.025055615231394768, 0.05547091364860535, -0.02376558445394039], [0.02587399072945118, -0.03435727208852768, 0.02927953563630581, -0.01063837856054306, 0.041126642376184464, -0.054884832352399826, 0.0420483723282814, 0.018462197855114937]], dtype=torch.float32)
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
