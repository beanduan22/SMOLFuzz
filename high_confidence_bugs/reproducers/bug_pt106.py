#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 106
Bug Type: inconsistent
Detail  : output[0]: l2=9.0167e-02 > threshold=1e-03 finite_elements=1 shape=[]
APIs    : torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.Dropout, torch.enable_grad
Mutation: scale_large

Run: python3 bug_pt106.py
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
        self.bn = nn.BatchNorm1d(8)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            x = self.fc1(x)
            y = torch.sin(x) * torch.cos(x)
            z = self.bn(y)
            w = self.drop(z)
            v = self.fc2(w)
            jac = torch.autograd.functional.jacobian(
                lambda t: torch.sin(self.fc1(t)), x
            )
        return jac.sum()

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
    x = torch.tensor([[119823.890625, -454772.75, 622313.875, 156369.203125, -92904.015625, 339488.09375, -284196.96875, 274293.8125], [164817.3125, -69768.0, -231042.875, 286670.78125, -460369.875, 76860.140625, -1131659.875, -973842.875], [215890.828125, -7169.603515625, -560793.3125, -393479.5625, 1604054.375, -836846.5, -413339.84375, 99432.796875], [-562458.4375, -671573.8125, 500004.09375, -1478818.625, 262910.59375, 590900.375, 196897.09375, -153577.109375]], dtype=torch.float32)
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
