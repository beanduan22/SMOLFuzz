#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 384
Bug Type: inconsistent
Detail  : output[0]: l2=5.0380e-02 > threshold=1e-03 finite_elements=1 shape=[]
APIs    : torch.nn.Linear, torch.nn.PReLU, torch.nn.BatchNorm1d, torch.nn.SyncBatchNorm
Mutation: scale_large

Run: python3 bug_pt384.py
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
        self.prelu = nn.PReLU()
        self.bn = nn.BatchNorm1d(8)
        self.sync_bn = nn.SyncBatchNorm(8)
        self.fc2 = nn.Linear(8, 8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = nn.functional.smooth_l1_loss(y, torch.zeros_like(y), reduction='mean')
            t = self.prelu(z + y)
            u = self.bn(t)
            v = self.sync_bn(u)
            w = torch.multiply(v, torch.sin(v))
            x_norm = w.norm(dim=1, keepdim=True)
            r = torch.divide(w, x_norm)
            s = torch.stft(r.squeeze(1), n_fft=8, return_complex=True)
            jacobian = torch.autograd.functional.jacobian(lambda t: torch.sin(self.fc2(t)), x)
        return jacobian.sum()

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
    x = torch.tensor([[432001.40625, -178943.828125, 256427.671875, 108605.03125, 622762.5, 254057.546875, -475429.4375, -316443.875], [1751907.375, -486254.625, 332341.28125, 38548.34765625, 1476846.75, 145677.0, 670375.875, -256612.796875], [-329658.375, -190186.09375, 829730.125, -541971.375, -258808.109375, 26393.611328125, 937689.9375, -442783.5625], [-57529.71875, 1021103.0, 439405.78125, 417694.65625, 58830.59375, -934694.0, 309999.1875, 103314.078125]], dtype=torch.float32)
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
