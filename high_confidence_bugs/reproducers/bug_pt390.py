#!/usr/bin/env python3
"""
SMOLFuzz Bug Reproducer — PyTorch
Model   : 390
Bug Type: inconsistent
Detail  : output[0]: l2=3.5405e-02 > threshold=1e-03 finite_elements=32 shape=[4, 8]
APIs    : torch.nn.Linear, torch.sin, torch.cos, torch.clip
Mutation: scale_large

Run: python3 bug_pt390.py
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
        self.softmin = nn.Softmin(dim=1)

    def forward(self, x):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = self.fc1(x)
            z = torch.sin(y) * torch.cos(y)
            u = torch.clip(z, min=-0.5, max=0.5)
            v = torch.log_(u.abs() + 1e-6)
            w = self.fc2(v)
        return self.softmin(w)

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
    x = torch.tensor([[45819.94921875, 435.01019287109375, -70678.9921875, -93307.0078125, 139226.09375, 128386.9140625, -21955.814453125, 164057.484375], [81980.7265625, 21167.39453125, -106764.3984375, 9196.94140625, 75262.8359375, 7774.115234375, 15653.3037109375, 12470.650390625], [48229.671875, -10403.791015625, -32509.845703125, 38698.37109375, -50140.59375, 5599.18115234375, -168723.453125, 135946.109375], [45057.29296875, 137189.703125, -52317.76171875, 108110.421875, 33284.98046875, -14167.20703125, -33159.5625, 2439.390869140625]], dtype=torch.float32)
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
