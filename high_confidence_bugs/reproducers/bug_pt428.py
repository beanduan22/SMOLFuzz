#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 428
Bug Type: inconsistent
Detail  : output[0]: l2=1.0558e-02 > threshold=1e-03 finite_elements=8 shape=[8]
APIs    : torch.nn.Linear, torch.nn.GELU, torch.sin, torch.cos
Mutation: scale_large

Run: python3 bug_pt428.py
Requires: PyTorch with CUDA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model ──────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(16, 8)
        self.sync_bn = nn.SyncBatchNorm(8)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.gelu(self.fc1(x))
            z = torch.sin(y) * torch.cos(y)
            w = self.fc2(z)
            b = self.sync_bn(w)
            c = torch.special.expm1(w)
            d = c.true_divide_(w)
            e = torch.aminmax(d, dim=0)[0]
        return e

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
    x = torch.tensor([[-586343.9375, -1134628.875, -1039412.3125, -486660.0, 911998.875, -39699.28125, -1014223.3125, -486190.8125], [114514.515625, -210469.015625, 508707.46875, -488173.71875, 1144397.0, 55448.6953125, 1246786.25, -761446.8125], [-903962.4375, 991748.6875, -833011.875, -225015.203125, -253670.859375, 225024.6875, -9477.9990234375, 300967.90625], [354369.65625, -1006495.3125, 865711.3125, 1399779.375, 357127.25, -509767.0, 424651.21875, -628146.5625]], dtype=torch.float32)
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
